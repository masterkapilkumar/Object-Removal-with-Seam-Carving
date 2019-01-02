# Object-Removal-with-Seam-Carving

## Object Removal

Removes all the seams passing through the object, decreasing the size of the resulting image.
```
usage: object_removal.py [-h] input_file template output_file

positional arguments:
  input_file   Input image
  template     Object template
  output_file  Output image

optional arguments:
  -h, --help   show this help message and exit
```


## Object Removal and Resizing

Removes all the seams passing through the object, and the insert high energy seams to resize image to the desired size

```
usage: seam_carving.py [-h] input_file output_file height width

positional arguments:
  input_file   Input image
  output_file  Output image
  height       Target height
  width        Target width

optional arguments:
  -h, --help   show this help message and exit
```
