from colorsys import hls_to_rgb, rgb_to_hls
from scipy import ndimage
from glob import glob
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import shlex
import tempfile
from colorthief import ColorThief
from flask import Flask, flash, jsonify, render_template, request, redirect, session, url_for, send_from_directory
from PIL import Image
import os, io
import rembg
import numpy as np
import torchvision
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageColor, ImageEnhance
from colorthief import ColorThief
import openai
# from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

# ---requirements for similarity detection----
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('clip-ViT-B-32')

# Flask app
app = Flask(__name__)

# Set allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Set upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'root'

# Set OpenAI API key and model_engine for Product Description
openai.api_key = "YOUR_API_KEY_HERE"
model_engine = "text-curie-001"

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    # Check if the file extension is allowed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model


def remove_bg_model(image_path):
    model = load_model()
    image = Image.open(image_path).convert('RGB')
    image_org = image.copy()
    image_sharp = image.filter(ImageFilter.SHARPEN)
    image_tensor = torchvision.transforms.functional.to_tensor(image_sharp)
    outputs = model([image_tensor])
    masks = outputs[0]['masks'].detach().numpy()
    labels = outputs[0]['labels'].detach().numpy()
    scores = outputs[0]['scores'].detach().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    for i in range(masks.shape[0]):
        masks[i, 0] = cv2.dilate(masks[i, 0], kernel)
    idxs = np.argsort(-scores)[:5]
    masks = masks[idxs]
    labels = labels[idxs]
    scores = scores[idxs]
    return image_org, masks, labels, scores


def remove_background(image_path):
    input_image = Image.open(session.get('image'))
    with open(image_path, "rb") as f:
        output_image = rembg.remove(f.read())

    object_image = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # Get the mask from the object image using remove_bg_model
    image_org, masks, labels, scores = remove_bg_model(image_path)

    mask = masks[0]
    mask = mask.squeeze()
    y_coords, x_coords = np.where(mask)
    mask_center_x = np.mean(x_coords)
    mask_center_y = np.mean(y_coords)
    # Calculate the center coordinates of the image
    image_center_x = input_image.size[0] / 2
    image_center_y = input_image.size[1] / 2

    # Calculate the distance to move the object horizontally and vertically
    dx = int(image_center_x - mask_center_x)
    dy = int(image_center_y - mask_center_y)

    # Move the object to the center of the image
    object_centered = Image.new('RGBA', object_image.size, (0, 0, 0, 0))
    object_centered.paste(object_image, (dx, dy))

    # Create a new image with a white background of the same size as the input image
    output_size = input_image.size
    output_image = Image.new('RGBA', output_size, (0, 0, 0, 0))
    
    # Paste the centered object image onto the white background
    output_image.paste(object_centered, (0, 0), object_centered)

    # Save the output image to a file and return the file path
    # output_path = 'output.png'
    # output_image.save(output_path)
    return output_image.convert("RGBA")


def add_default_background(image):    
    # Get the dominant color of the image using ColorThief
    with tempfile.NamedTemporaryFile(delete=False) as f:
        image.save(f, 'PNG')
        f.flush()
        f.close()
        color_thief = ColorThief(f.name)
        
    dominant_color = color_thief.get_color(quality=1)
    with_alpha = image.convert('RGBA')

    # Convert the main color from RGB to HSL
    main_color_hls = rgb_to_hls(*[c/255.0 for c in dominant_color])

    # Adjust the Hue by 180 degrees to get the complementary color
    bg_color_hls = (main_color_hls[0] + 0.5) % 1.0
    bg_color_rgb = np.array(hls_to_rgb(bg_color_hls, main_color_hls[1], main_color_hls[2])) * 255
    bg_color = tuple(bg_color_rgb.astype('int64'))

    background = Image.new('RGBA', with_alpha.size, bg_color)
    final_image = Image.alpha_composite(background, with_alpha)
    final_image = final_image.convert('RGB')  # Convert to RGB before saving as JPEG
    # Enhance the final image
    sharpness_factor = 1.5  # Increase the sharpness
    final_image = ImageEnhance.Sharpness(final_image).enhance(sharpness_factor)
    
    return final_image


def blur_detection(filepath):
    # Load the image using OpenCV
    img = cv2.imread(filepath)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Compute the variance of the Laplacian
    variance = laplacian.var()
    variance = round(variance,2)
    # Define a threshold value for the variance
    threshold = 100
    
    # Check if the variance is below the threshold
    if variance < threshold:
        return 'Blurry (variance of {variance})'.format(variance=variance)
    else:
        return 'Not blurry (variance of {variance})'.format(variance=variance)
    

def detect_similarity(filepath1, filepath2):
    # Load images
    img1 = Image.open(filepath1).convert('RGB')
    img2 = Image.open(filepath2).convert('RGB')

    # Encode images
    img1_encoded = model.encode(img1, convert_to_tensor=True)
    img2_encoded = model.encode(img2, convert_to_tensor=True)

    # Compute cosine similarity between the embeddings
    cos_sim = util.pytorch_cos_sim(img1_encoded, img2_encoded).item()

    # Convert similarity score to percentage
    percentage_similarity = round(cos_sim * 100, 2)

    return percentage_similarity


# Define the function to generate product descriptions
def generate_product_description(keywords):
    keyword_string = " ".join(keywords)
    prompt = f"Generate a product description using {keyword_string} keywords"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )
    description = response.choices[0].text.strip()
    return description


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    # Get the uploaded file
    file = request.files['image']

    # Check if the file has a valid extension
    if not allowed_file(file.filename):
        return "Invalid file extension. Allowed extensions are: {}".format(", ".join(app.config['ALLOWED_EXTENSIONS']))

    # Save the uploaded file to disk
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['image'] = filepath

    # Remove the background from the image using rembg
    try:
        output_image = remove_background(filepath)
    except Exception as e:
        return f"Error removing background: {str(e)}"

    # Save the image with transparent background to disk
    object_filename = f'object_{filename}'
    object_filepath = os.path.join(app.config['UPLOAD_FOLDER'], object_filename)
    output_image.save(object_filepath,'PNG')
    session['object'] = object_filepath

    # Add a white background to the image
    final_image = add_default_background(output_image)

    # Save the image with transparent background to disk
    defaultBg_filename = f'defaultBg_{filename}'
    defaultBg_filepath = os.path.join(app.config['UPLOAD_FOLDER'], defaultBg_filename)
    final_image.save(defaultBg_filepath)

    # Display the image to the user for approval
    size = final_image.size
    return render_template('approve.html', filename=defaultBg_filename, size=size)


@app.route('/approve/<filename>', methods=['POST'])
def approve(filename):
    # Get the user's choice
    choice = request.form.get('choice')

    if choice == 'yes':
        # Render a new HTML template for the user to choose whether to change background
        return render_template('change_background.html', filename=filename)

    else:
        defaultBg_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.remove(defaultBg_filepath)
        # Display the original uploaded image
        original_filename = filename.split('_', 1)[1]
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        img = Image.open(original_filepath)

        size = img.size
        return render_template('select_roi.html', filename=original_filename, size=size)


@app.route('/uploads/<filename>')
def send_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename) 


@app.route('/change_background/<filename>', methods=['POST'])
def change_background(filename):
    # Get the user's choice
    choice = request.form.get('choice')
    
    if choice == 'yes':
        # Render a new HTML template for the user to choose the background color
        original_filename = filename.split('_', 1)[1]
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        img = Image.open(original_filepath)
        size = img.size
        return render_template('select_background.html', filename=filename, size=size)

    else:
        # Save the final file
        final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = Image.open(final_filepath)
        img.save(final_filepath)

        # remove the object file
        remove_file_path = session.get('object')
        os.remove(remove_file_path)
        
        return render_template('send_file.html', filename=filename)
        
        
@app.route('/background/<filename>', methods=['POST'])
def background(filename):
    # Get the user's selected background color

    object2 = session.get('object')   
    color = request.form.get('color')     
    img = Image.open(object2)
    
    if color:
        background_color = ImageColor.getrgb(color)
       
    else:
        # Use the default white background color
        background_color = (255, 255, 255)

    # Create a new image with the same size as the original and fill it with the desired background color
    bg_img = Image.new(mode="RGBA", size=img.size, color=background_color)
   
    # Merge the original image with the new background
    final_img = Image.alpha_composite(bg_img, img)

    # Save the modified image to disk
    size = final_img.size
    filename1 = filename.split('_', 1)[1]
    img_filename = f'editedBg_{filename1}'
    img_filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    final_img = final_img.convert('RGB')  # Convert to RGB before saving as JPEG
   
    final_img.save(img_filepath)

    # remove the object file
    remove_file_path = session.get('object')
    os.remove(remove_file_path)

    # Display the final image to the user
    return render_template('send_file.html', filename=img_filename)


@app.route('/process_roi/<filename>', methods=['POST'])
def process_roi(filename):
    # Get the selected ROI
    x = float(request.form.get('x'))
    y = float(request.form.get('y'))
    width = float(request.form.get('width'))
    height = float(request.form.get('height'))
    
    # Open the original image
    original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(original_filepath)
    
    # Crop the ROI from the original image
    roi = img.crop((x-30, y-30, x+width, y+height))
  
    # Save the ROI to a temporary file
    roi_temp_file = os.path.join(app.config['UPLOAD_FOLDER'], 'roi_temp.png')
    roi.save(roi_temp_file)

    # Remove the background from the ROI using rembg
    try:
        image_org, masks, labels, scores = remove_bg_model(roi_temp_file)
        draw = ImageDraw.Draw(roi)
        for i, mask in enumerate(masks):
            mask = np.array(mask[0])
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            draw.bitmap((0,0), mask_img, fill='gray')
        
        selected_mask = masks[0]
        # Convert the mask to a binary mask
        mask = selected_mask.squeeze()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.point(lambda x: 255 if x > 128 else 0, mode='1')

        # Get the object from the original image
        object = image_org.copy()
        object.putalpha(mask_img)
       
        enhancer = ImageEnhance.Color(object)
        object = enhancer.enhance(1.5)
        object2 = rembg.remove(object)

        object_filename = f'object_{filename}'
        object_filepath = os.path.join(app.config['UPLOAD_FOLDER'], object_filename)
        
        # Get the center of mass of the mask
        mask_center = ndimage.measurements.center_of_mass(mask)

        # Get the size of the input image
        input_size = img.size

        # Calculate the center of the input image
        input_center = (input_size[0]/2, input_size[1]/2)

        # Calculate the offset to align the object at the center of the input image
        offset = (int(input_center[0] - mask_center[1]), int(input_center[1] - mask_center[0]))

        # Create a new transparent image with the size of the input image
        transparent_background = Image.new("RGBA", input_size, (0, 0, 0, 0))

        # Paste the object onto the transparent image at the center
        transparent_background.paste(object2, offset, object2)

        transparent_background.save(object_filepath,'PNG')
        session['object'] = object_filepath

        # Save the final image with the object on a transparent background
        final_image = transparent_background.convert('RGB')
        final_image_filename = f'final_{filename}'
        final_image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_image_filename)
        final_image.save(final_image_filepath)
        output_image=transparent_background
        # output_roi = remove_background(roi_temp_file)
    except Exception as e:
        return f"Error removing background: {str(e)}"
    
    # Add a white background to the ROI
    final_roi = add_default_background(output_image)
    
    # Save the image with the selected ROI to disk
    roi_filename = f'roiBg_{filename}'
    roi_filepath = os.path.join(app.config['UPLOAD_FOLDER'], roi_filename)
    final_roi.save(roi_filepath)

    # Delete the temporary file
    os.remove(roi_temp_file)
    os.remove(final_image_filepath)

    # Display the image to the user for approval
    size = final_roi.size

    return render_template('approve.html', filename=roi_filename, size=size)


@app.route('/blur_detection', methods=['POST'])
def blur_detection_view():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    # Check if the file has a valid extension
    if not allowed_file(file.filename):
        return "Invalid file extension. Allowed extensions are: {}".format(", ".join(app.config['ALLOWED_EXTENSIONS']))
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
    except Exception as e:
        error_message = f"Error while saving file: {e}"
        return jsonify({'error': error_message}), 500
    blur_result = blur_detection(filepath)
    
    return render_template('blur_detection.html', filepath=filepath, blur_result=blur_result)


@app.route('/similarity_measure', methods=['POST'])
def similarity_measure():
    # Check if the uploads folder exists and has any files
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir) or not os.listdir(upload_dir):
        return "No files found in uploads folder"
    
    if 'image' not in request.files:    
        return redirect(url_for('index'))

    # Get the uploaded image file
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Check if the file has a valid extension
    if not allowed_file(file.filename):
        return "Invalid file extension. Allowed extensions are: {}".format(", ".join(app.config['ALLOWED_EXTENSIONS']))

    # Save the file to disk
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Compute the similarity between the input image and all other images in the folder
    similarities = []
    for file_path in glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        if file_path != filepath:
            similarity = detect_similarity(filepath, file_path)
            filename = os.path.basename(file_path)
            similarities.append((similarity, filename, file_path))

    # Sort the results by the similarity score to get the top 5 scored images
    similarities.sort(reverse=True)
    top5_similarities = similarities[:5]
   
    # Render the similarity_measure.html template with the similarity percentage, input image path,
    # top 5 scored images paths, and the image path that scored maximum similarity with the input image
    return render_template('similarity_measure.html',
                           similarity_percentage=top5_similarities[0][0],
                           filepath_input=filepath,
                           top5_similarities=top5_similarities[:5])


@app.route("/generate_description", methods=["POST"])
def product_description():
    # Get the keywords from the form submission
    # keywords = [kw.strip() for kw in request.form["keywords"].split(",")]
    keywords = shlex.split(request.form["keywords"])
    # Remove the comma from the end of each keyword
    keywords = [keyword.rstrip(',') for keyword in keywords]
    # Generate the product description using the keywords
    description = generate_product_description(keywords)
    # Render the description template with the generated description
    return render_template("description.html", description=description)


if __name__ == '__main__':
    app.run(debug=True)
