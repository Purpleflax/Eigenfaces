import numpy as np
from PIL import Image

def load_images():
    images = []
    for i in range(1, 41):
        for j in range(1, 11):
            image_path = f'att_faces/s{i}/{j}.pgm'
            image = Image.open(image_path)
            image = np.array(image, dtype=np.float64).flatten()
            images.append(image)
    return np.array(images)

def create_image_matrix(images):
    image_matrix = np.array(images)
    image_matrix = image_matrix.reshape(image_matrix.shape[0], -1)
    return image_matrix

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

if __name__ == '__main__':
    images = load_images()
    image_matrix = create_image_matrix(images)
    eigenfaces, mean_face = pca(image_matrix)
    
    # Precompute weights for all training faces
    training_weights = [np.dot(image_matrix[i, :] - mean_face, eigenfaces) for i in range(image_matrix.shape[0])]
    
    input_face = image_matrix[21]  # Use the first face in the dataset as the input
    recognized_id = recognize_face(input_face, eigenfaces, mean_face, training_weights)
    print('Recognized ID:', recognized_id)
