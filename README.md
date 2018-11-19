# face-recognition
Face recognition on LFW dataset using a pretrained model from OpenFace project. The project was built using two existing projects:

- https://medium.freecodecamp.org/making-your-own-face-recognition-system-29a8e728107c
- https://krasserm.github.io/2018/02/07/deep-face-recognition/


# Usage

```./get_big_files.sh```

Downloads and unpacks image data (LFW dataset), face recognition data and the pretrained model.

```python2 ./face_recognition.py batch pairs.txt```

Compares and saves the distances in every pair in the .txt file while indicating if two images are a match or not.

```python2 ./face_recognition.py compare face1.jpg face2.jpg```

Compares two faces in the LFW dataset. Arguments are the names of the files.

```python2 ./find_optimal_threshold.py pairs_out.txt```

Calculates F1 score and finds the optimal threshold from distance data.