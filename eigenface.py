# Import the necessary libraries
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, Label, Toplevel
from tkinter.ttk import Style
import threading
import os
from PIL import Image

# Function to load the images from the att_faces dataset
def load_images():
    images = []
    num_subjects = len(os.listdir('att_faces'))
    for i in range(1, num_subjects):
        print(f"Loading subject {i}...")
        for j in range(1, 11):
            image_path = f'att_faces/s{i}/{j}.pgm'
            image = Image.open(image_path)
            image = np.array(image, dtype=np.float64).flatten()
            images.append(image)
    return np.array(images)

# Function to convert the images into a matrix representation for PCA
def create_image_matrix(images):
    image_matrix = np.array(images)
    image_matrix = image_matrix.reshape(image_matrix.shape[0], -1)
    return image_matrix

### PCA and Face Recognition Functions ###
# PCA function to compute the eigenfaces and mean face
# Returns the eigenfaces and mean face
def pca(X):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    # Compute the smaller matrix (L)
    L = np.dot(X_centered, X_centered.T)
    eigenvalues, eigenvectors_L = np.linalg.eigh(L)
    # Sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_L = eigenvectors_L[:, idx]
    # Compute the eigenvectors for the high-dimensional space
    eigenvectors = np.dot(X_centered.T, eigenvectors_L)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)  # Normalize the eigenvectors
    return eigenvectors, mean_face

# Function to recognize a face using the eigenfaces by taking the input face and comparing it to the training data
# Returns the subject ID of the recognized face
def recognize_face(input_face, eigenfaces, mean_face, training_weights):
    input_face_centered = input_face - mean_face
    input_weights = np.dot(input_face_centered, eigenfaces)

    min_distance = np.inf
    min_distance_id = -1

    for i, training_weight in enumerate(training_weights):
        distance = np.linalg.norm(input_weights - training_weight)

        if distance < min_distance:
            min_distance = distance
            min_distance_id = i

    subject_id = min_distance_id // 10 + 1
    return subject_id

# Convert to .pgm format
def pre_process_image(path):
    image = Image.open(path)
    image = image.convert('L')
    image = image.resize((92, 112), Image.ANTIALIAS)
    image = np.array(image, dtype=np.float64).flatten()
    return image

# Function to add a new subject, takes an array of images and adds a new subject
# with the ID of the last subject + 1
def add_subject(images, new_subject_image_paths):
    if len(new_subject_image_paths) != 10:
        raise ValueError("Each subject must have 10 images.")
    
    new_subject_images = np.array([pre_process_image(path) for path in new_subject_image_paths])
    return np.concatenate((images, new_subject_images))

# Function to save the processed images in the att_faces directory
def update_status(message):
    status_label.config(text=message)
    root.update()

# Function to recognize a face using the eigenfaces in a separate thread
def threaded_recognition(image_matrix, eigenfaces, mean_face, training_weights):
    update_status("Recognizing face...")
    recognized_id = recognize_face(image_matrix, eigenfaces, mean_face, training_weights)
    result_label.config(text=f"Recognized ID: {recognized_id}")
    update_status("Ready")

# Function to upload an image and recognize the face
def upload_action(event=None):
    filename = filedialog.askopenfilename()
    if not filename:
        return
    update_status("Processing image...")
    image = pre_process_image(filename)
    image_matrix = image.reshape((1, -1))
    threading.Thread(target=threaded_recognition, args=(image_matrix, eigenfaces, mean_face, training_weights)).start()

# Function to add a new subject in a separate thread
def threaded_add_subject(foldername, eigenfaces, mean_face):
    update_status("Adding new subject...")
    image_paths = [os.path.join(foldername, f) for f in os.listdir(foldername) if f.endswith('.jpg')]
    image_paths = sorted(image_paths)  # Sort to maintain order

    new_subject_images = np.array([pre_process_image(path) for path in image_paths])
    save_processed_images(foldername, new_subject_images)  # Save the processed images
    
    global images
    images = add_subject(images, image_paths)
    global training_weights
    training_weights = [np.dot(images[i, :] - mean_face, eigenfaces) for i in range(images.shape[0])]

    update_status("New subject added. Ready.")

# Function to add a new subject
def add_subject_action(event=None):
    foldername = filedialog.askdirectory()
    if not foldername:
        return
    threading.Thread(target=threaded_add_subject, args=(foldername, eigenfaces, mean_face)).start()

# Function to save the processed images in the att_faces directory
def save_processed_images(foldername, new_subject_images):
    # Determine the new subject ID based on existing directories
    subject_ids = [int(d.split('s')[1]) for d in os.listdir('att_faces') if os.path.isdir(os.path.join('att_faces', d))]
    new_subject_id = max(subject_ids) + 1 if subject_ids else 1
    
    # Create a new directory for the subject
    new_subject_dir = os.path.join('att_faces', f's{new_subject_id}')
    os.makedirs(new_subject_dir, exist_ok=True)
    
    # Save each image in the PGM format
    for i, image in enumerate(new_subject_images, start=1):
        image_path = os.path.join(new_subject_dir, f'{i}.pgm')
        Image.fromarray(image.reshape(112, 92)).convert('L').save(image_path)
        
# Tooltip class
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     # miliseconds
        self.wraplength = 180   # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # Creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      background="#ffffff", relief='solid', borderwidth=1,
                      wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()
            
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Eigenface Recognizer")
    root.configure(bg="#f0f0f0")
    root.geometry("600x350")
    
    images = load_images()
    image_matrix = create_image_matrix(images)
    eigenfaces, mean_face = pca(image_matrix)
    training_weights = [np.dot(images[i, :] - mean_face, eigenfaces) for i in range(images.shape[0])]

    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 12, 'bold'), borderwidth='4', foreground='black')
    style.configure('TLabel', font=('Helvetica', 10), background='#f0f0f0')
    style.map('TButton', foreground=[('active', '!disabled', 'green')], background=[('active', 'black')])

    header = tk.Label(root, text="Eigenface Recognizer", font=("Helvetica", 18, "bold"), bg='#f0f0f0')
    header.pack(pady=10)

    description = "This application uses Eigenfaces for face recognition. You can add new subjects or test the recognition with an existing dataset."
    description_label = tk.Label(root, text=description, wraplength=500, justify="center", bg='#f0f0f0')
    description_label.pack(pady=10)

    frame = tk.Frame(root, bg='#f0f0f0')
    frame.pack(pady=20)

    upload_button = tk.Button(frame, text="Upload an Image", command=upload_action)
    upload_button.grid(row=0, column=0, padx=20)
    CreateToolTip(upload_button, "Upload an image to recognize the subject.")

    add_subject_button = tk.Button(frame, text="Add New Subject", command=add_subject_action)
    add_subject_button.grid(row=0, column=1, padx=20)
    CreateToolTip(add_subject_button, "Add a new subject to the dataset.\n\nEach subject must have 10 images.\n\n" +
                                       "Images must be in JPG format.\n\nImages must focus on the subject's face from the front.\n\n" +
                                       "The frame should be almost entirely filled by the face.")

    status_label = tk.Label(root, text="Ready", fg="blue", bg='#f0f0f0')
    status_label.pack(pady=5)

    result_label = tk.Label(root, text="", bg='#f0f0f0')
    result_label.pack(pady=10)

    root.mainloop()