import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt
import os

# Load the image
img = cv.imread(r'C:\Users\RayaBit\Desktop\Face-detection\Abba_Eban_0001.jpg', -1)
img_Gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Load the Haar cascade classifier for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(img_Gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected faces and save each face as a separate image
face_count = 0
for (x, y, w, h) in faces:
    cv.rectangle(img_Gray, (x, y), (x+w, y+h), (100, 255, 0), 2)
    
    # Extract the face ROI
    face_roi = img_Gray[y:y+h, x:x+w]
    
    # Save the face ROI
    face_roi_path = r'C:\\Users\\RayaBit\\Desktop\\Face-detection\\Abba_Eban_0001_face_.jpg'
    # face_roi_path = f'C:\\Users\\RayaBit\\Desktop\\Abba_Eban_0001_face_{face_count}.jpg'
    cv.imwrite(face_roi_path, face_roi)
    # face_count += 1

# Display the image with detected faces
cv.imshow('Face Detection', img_Gray)
cv.imshow('ROI Area', face_roi)

# Save the image with detected faces
output_path = r'C:\Users\RayaBit\Desktop\Face-detection\\Abba_Eban_0001_detected.jpg'
cv.imwrite(output_path, img_Gray)

# Wait for a key press and close the image window
cv.waitKey(0)
cv.destroyAllWindows()

def divide_image_into_cells(image_path, rows, cols):
    # Load the image
    image = cv.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None, None, None

    # Get image dimensions
    height, width, _ = image.shape

    # Calculate the size of each cell
    cell_height = height // rows
    cell_width = width // cols

    cells = []

    # Loop through the image and extract each cell
    for i in range(rows):
        row_cells = []
        for j in range(cols):
            start_row = i * cell_height
            end_row = (i + 1) * cell_height
            start_col = j * cell_width
            end_col = (j + 1) * cell_width

            cell = image[start_row:end_row, start_col:end_col]
            row_cells.append(cell)

        cells.append(row_cells)

    return cells, cell_height, cell_width

def save_cells(cells, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each cell as an individual image
    for i, row_cells in enumerate(cells):
        for j, cell in enumerate(row_cells):
            cell_filename = os.path.join(output_dir, f'cell_{i}_{j}.jpg')
            cv.imwrite(cell_filename, cell)
            print(f'Saved {cell_filename}')

def create_image_with_dashes(cells, cell_height, cell_width):
    rows = len(cells)
    cols = len(cells[0])

    # Determine the size of the new image
    dash_size = 2  # size of the dash line
    new_height = rows * cell_height + (rows - 1) * dash_size
    new_width = cols * cell_width + (cols - 1) * dash_size

    # Create a new image with white background
    new_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

    for i in range(rows):
        for j in range(cols):
            start_row = i * (cell_height + dash_size)
            start_col = j * (cell_width + dash_size)

            new_image[start_row:start_row + cell_height, start_col:start_col + cell_width] = cells[i][j]

    # Add dashes
    for i in range(1, rows):
        new_image[i * (cell_height + dash_size) - dash_size:i * (cell_height + dash_size), :] = 0

    for j in range(1, cols):
        new_image[:, j * (cell_width + dash_size) - dash_size:j * (cell_width + dash_size)] = 0

    return new_image

def display_image(image):
    # Convert the BGR image to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Display the image
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide the axes
    plt.show()

def save_image(image, output_path):
    cv.imwrite(output_path, image)
    print(f'Saved image to {output_path}')

# Example usage
image_path = r'C:\\Users\\RayaBit\\Desktop\\Face-detection\\Abba_Eban_0001_face_.jpg'  # Replace with your image path
output_dir = r'C:\Users\RayaBit\Desktop\Face-detection\cell.jpg'  # Directory to save individual cells
output_image_path = r'C:\Users\RayaBit\Desktop\Face-detection\figure1.jpg'  # Replace with your desired output image path
rows, cols = 7, 7
cells, cell_height, cell_width = divide_image_into_cells(image_path, rows, cols)

if cells:
    save_cells(cells, output_dir)
    dashed_image = create_image_with_dashes(cells, cell_height, cell_width)
    display_image(dashed_image)
    save_image(dashed_image, output_image_path)

def lbp_image(image2):
    # Convert to grayscale if the image is colored
    if len(image2.shape) == 3:
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    # Define the LBP operation
    def lbp_pixel(pixel, neighbors):
        binary = ''.join(['1' if neighbor >= pixel else '0' for neighbor in neighbors])
        return int(binary, 2)
    
    # Apply LBP to each pixel
    lbp_img = np.zeros_like(image2)
    rows, cols = image2.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            neighbors = [
                image2[i-1, j-1], image2[i-1, j], image2[i-1, j+1],
                image2[i, j+1], image2[i+1, j+1], image2[i+1, j],
                image2[i+1, j-1], image2[i, j-1]
            ]
            lbp_img[i, j] = lbp_pixel(image2[i, j], neighbors)
    return lbp_img

# Load image
image_path2 = r'C:\\Users\\RayaBit\\Desktop\\Face-detection\\Abba_Eban_0001_face_.jpg'
image2 = cv.imread(image_path2)

# Compute the LBP image
lbp_img = lbp_image(image2)

#Display LBP image
cv.imshow('LBP Img', lbp_img)

#Saving The image
cv.imwrite(r'C:\\Users\\RayaBit\\Desktop\\Face-detection\\LBP_Img.jpg', lbp_img)

def divide_image_into_cells(image, num_rows, num_cols):
    # Get image dimensions
    height, width, _ = image.shape
    
    # Calculate cell size
    cell_height = height // num_rows
    cell_width = width // num_cols
    
    cells = []
    
    for r in range(num_rows):
        for c in range(num_cols):
            # Calculate cell boundaries
            y1 = r * cell_height
            y2 = (r + 1) * cell_height
            x1 = c * cell_width
            x2 = (c + 1) * cell_width
            
            # Extract cell from the image
            cell = image[y1:y2, x1:x2]
            cells.append(cell)
            
    return cells

def save_cells(cells, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save each cell as a separate image in the output folder
    for i, cell in enumerate(cells):
        cell_path = os.path.join(output_folder, f'cell_{i}.jpg')
        cv.imwrite(cell_path, cell)
        print(f'Saved cell {i} to {cell_path}')

def display_grid_with_dashes(cells, num_rows, num_cols):
    # Create a blank canvas to display the grid of cells
    canvas_height = cells[0].shape[0] * num_rows
    canvas_width = cells[0].shape[1] * num_cols
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Populate the canvas with cells and add dashes
    for r in range(num_rows):
        for c in range(num_cols):
            cell = cells[r * num_cols + c]
            canvas[r * cell.shape[0]:(r + 1) * cell.shape[0], c * cell.shape[1]:(c + 1) * cell.shape[1]] = cell
            
            # Add vertical dash lines
            if c < num_cols - 1:
                canvas[:, (c + 1) * cell.shape[1] - 1:(c + 1) * cell.shape[1] + 1] = [0, 0, 0]
                
        # Add horizontal dash lines
        if r < num_rows - 1:
            canvas[(r + 1) * cell.shape[0] - 1:(r + 1) * cell.shape[0] + 1, :] = [0, 0, 0]

    # Display and save the canvas
    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    plt.title(f'Grid of {num_rows}x{num_cols} Cells with Dashes')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('grid_with_dashes.jpg')
    plt.show()

# Example usage
image_path = r'C:\\Users\\RayaBit\\Desktop\\Face-detection\\LBP_Img.jpg'  # Replace with your image path
output_folder = r'C:\\Users\\RayaBit\\Desktop\\Face-detection\\LBP_Img_folder.jpg'  # Replace with your output folder path for saving cells

# Load the image
image = cv.imread(image_path)

# Divide the image into cells
num_rows = 7
num_cols = 7
cells = divide_image_into_cells(image, num_rows, num_cols)

# Save each cell as a separate image in the output folder
save_cells(cells, output_folder)

# Display and save the grid with dashes
display_grid_with_dashes(cells, num_rows, num_cols)


def calculate_histogram(image):
    # Convert image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Calculate histogram using OpenCV
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist.flatten()

def read_images_and_calculate_histograms(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    histograms = []

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {filename}. Skipping...")
                continue

            # Calculate histogram for the image
            hist = calculate_histogram(image)
            histograms.append(hist)

            # Display and save histogram plot
            plt.figure()
            plt.plot(hist, color='black')
            plt.title(f'Histogram of {filename}')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()

            histogram_image_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_histogram.jpg')
            plt.savefig(histogram_image_path)
            plt.close()

            print(f'Saved histogram of {filename} to {histogram_image_path}')

    return histograms


def plot_connected_histograms(histograms):
    # Plot all histograms together
    plt.figure(figsize=(12, 6))
    for i, hist in enumerate(histograms):
        plt.plot(hist, label=f'Image {i}')

    plt.title('Connected Histograms of Images')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
input_folder = r'C:\\Users\\RayaBit\\Desktop\\Face-detection\\LBP_Img_folder.jpg'  # Replace with your input folder containing images
output_folder = r'C:\Users\RayaBit\Desktop\Face-detection\histogram.jpg'  # Replace with your output folder for histograms
histograms = read_images_and_calculate_histograms(input_folder, output_folder)

if histograms:
    plot_connected_histograms(histograms)

def read_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def compute_3d_histogram(image):
    hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()
    return hist

def chi_square_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

def process_folder(input_folder):
    images, filenames = read_images_from_folder(input_folder)
    histograms = [compute_3d_histogram(img) for img in images]
    return histograms, filenames

def knn_recognition(test_image, training_histograms, training_filenames):
    test_hist = compute_3d_histogram(test_image)
    min_distance = float('inf')
    recognized_filename = None
    
    for hist, filename in zip(training_histograms, training_filenames):
        dist = chi_square_distance(test_hist, hist)
        if dist < min_distance:
            min_distance = dist
            recognized_filename = filename
    
    return recognized_filename, min_distance

# Example usage
input_folder = r'C:\Users\RayaBit\Desktop\Face-detection\Train'  # Replace with your input folder path
test_image_path = r'C:\Users\RayaBit\Desktop\Face-detection\Abba_Eban_0001.jpg'  # Replace with your test image path

# Process training images
training_histograms, training_filenames = process_folder(input_folder)

# Read and recognize test image
test_image = cv.imread(test_image_path, cv.IMREAD_COLOR)
if test_image is not None:
    recognized_filename, confidence = knn_recognition(test_image, training_histograms, training_filenames)
    print(f"The test image is recognized as: {recognized_filename} with confidence {confidence}")
else:
    print("Test image not found.")
