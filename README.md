# Eigenfaces Facial Recognition

## Overview

This project implements a facial recognition system using the eigenfaces method. It employs linear algebra principles, notably Principal Component Analysis (PCA), to project facial images onto a subspace spanned by a set of basis images known as eigenfaces. This technique streamlines the recognition process by capturing the most significant features of faces.

## Features

- Image loading and preprocessing from a dataset.
- Matrix representation of image data for PCA.
- Eigenfaces computation to form a basis set for facial representation.
- User interface for adding new faces to the system and for facial recognition.

## How to Use

1. Clone the repository.
2. Ensure you have the required Python packages installed: `numpy`, `tkinter`, and `PIL`.
3. Run the script with Python to launch the application.
4. Follow the UI prompts to either add your face to the database or perform facial recognition.

## Prerequisites

- Python 3.x
- NumPy
- PIL
- Tkinter

## Installation

To set up the project environment, install the necessary Python packages using pip:

```bash
pip install numpy pillow
```

## Usage
1. To start the application, execute the eigenface.py script.
2. To add a face to the database, select the directory with your images.
3. To recognize a face, upload an image, and the program will compare it to the database.

## Contributing
Contributions to the project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Acknowledgments
- L. Sirovich and M. Kirby for the foundational paper on eigenfaces.
- Matthew Turk and Alex Pentland for their research on face recognition using eigenfaces.
- The creators of the AT&T Faces Dataset used for training and testing the algorithm.

## Author
Jeffrey Reynolds