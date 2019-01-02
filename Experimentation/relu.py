import numpy as np
import cv2
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

out_image = None
img_counter = 0

def save_result(img, filename):
    cv2.imwrite(filename, img.astype(np.uint8))

def cumulative_map_backward(energy_map):
    output = np.copy(energy_map)
    m, n = energy_map.shape
    for row in range(1, m):
        for col in range(n):
            if(col==0):
                output[row, col] = energy_map[row, col] + np.amin(output[row - 1, :2])
            elif(col+3>n):
                output[row, col] = energy_map[row, col] + np.amin(output[row - 1, col - 1: n - 1])
            else:
                output[row, col] = energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
    return output

def delete_seam(seam_idx):
    global out_image
    output = np.zeros((out_image.shape[0], out_image.shape[1] - 1, 3))
    for row in range(out_image.shape[0]):
        for i in range(3):
            output[row, :, i] = np.delete(out_image[row, :, i], [seam_idx[row]])
    out_image = np.copy(output)

def add_seam(seam_idx):
    
    global out_image
    m, n = out_image.shape[: 2]
    output = np.zeros((m, n + 1, 3))
    for row in range(m):
        col = seam_idx[row]
        if col != 0:
            p = np.average(out_image[row, col - 1: col + 1, 0])
            output[row, col, 0] = p
            p = np.average(out_image[row, col - 1: col + 1, 1])
            output[row, col, 1] = p
            p = np.average(out_image[row, col - 1: col + 1, 2])
            output[row, col, 2] = p
            output[row, : col, 0] = out_image[row, : col, 0]
            output[row, : col, 1] = out_image[row, : col, 1]
            output[row, : col, 2] = out_image[row, : col, 2]
        else:
            p = np.average(out_image[row, col: col + 2, 0])
            output[row, col + 1, 0] = p
            p = np.average(out_image[row, col: col + 2, 1])
            output[row, col + 1, 1] = p
            p = np.average(out_image[row, col: col + 2, 2])
            output[row, col + 1, 2] = p
            output[row, col, 0] = out_image[row, col, 0]
            output[row, col, 1] = out_image[row, col, 1]
            output[row, col, 2] = out_image[row, col, 2]
        output[row, col + 1:, 0] = out_image[row, col:, 0]
        output[row, col + 1:, 1] = out_image[row, col:, 1]
        output[row, col + 1:, 2] = out_image[row, col:, 2]
    out_image = np.copy(output)

def find_seam(cumulative_map):
    m, n = cumulative_map.shape
    output = np.zeros((m,), dtype=np.uint32)
    output[m-1] = np.argmin(cumulative_map[-1])
    for row in range(m - 2, -1, -1):
        if output[row + 1] == 0:
            output[row] = np.argmin(cumulative_map[row, : 2])
        elif output[row + 1] > n - 3:
            output[row] = np.argmin(cumulative_map[row, output[row + 1]-1: n - 1]) + output[row + 1] - 1
        else:
            output[row] = np.argmin(cumulative_map[row, output[row + 1]-1: output[row + 1] + 2]) + output[row + 1] - 1
    return output

def calc_energy_map():
	image = out_image.copy()
	b, g, r = cv2.split(image)
	density = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1)) \
            + np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1)) \
            + np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
	cv2.normalize(density, density, -255, 255, cv2.NORM_MINMAX)
	return (density.clip(min=0)) 

def seams_insertion(num_pixel, rotation=False):
    
    global out_image
    global img_counter
    temp_image = np.copy(out_image)
    seams_record = []
    
    for i in range(num_pixel):
        energy_map = calc_energy_map()
        cumulative_map = cumulative_map_backward(energy_map)
        seam_idx = find_seam(cumulative_map)
        seams_record.append(seam_idx)
        delete_seam(seam_idx)

    out_image = np.copy(temp_image)
    n = len(seams_record)
    for i in range(n):
        cur_seam = seams_record.pop(0)
        add_seam(cur_seam)
        if(rotation):
            save_result(rotate_image(color_seam(cur_seam), 0).astype('uint8'), "out/"+str(img_counter)+".jpg")
        else:
            save_result(color_seam(cur_seam).astype('uint8'), "out/"+str(img_counter)+".jpg")
        img_counter+=1
        
        #shift seams to right
        for seam in seams_record:
            seam[np.where(seam >= cur_seam)] += 2

def filter(mat, kernel):
    return np.absolute(cv2.filter2D(mat, -1, kernel=kernel))

def cumulative_map_forward(energy_map):
    # kernel for forward energy map calculation
    kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
    kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
    kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
    
    b, g, r = cv2.split(out_image)
    #neighbour matrices
    matrix_x = filter(b, kernel_x) + filter(g, kernel_x) + filter(r, kernel_x)
    matrix_y_right = filter(b, kernel_y_right) + filter(g, kernel_y_right) + filter(r, kernel_y_right)
    matrix_y_left = filter(b, kernel_y_left) + filter(g, kernel_y_left) + filter(r, kernel_y_left)

    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            e_up, e_right, e_left = 10**20, 10**20, 10**20
            if col == 0:
                e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
            elif col == n - 1:
                e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
            else:
                e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
            e_up = output[row - 1, col] + matrix_x[row - 1, col]
            output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
    return output

def rotate_image(image, ccw):
    m, n, ch = image.shape
    output = np.zeros((n, m, ch))
    if ccw:
        image_flip = np.fliplr(image)
        for row in range(m):
            for c in range(ch):
                output[:, row, c] = image_flip[row, :, c]
    else:
        for row in range(m):
            for c in range(ch):
                output[:, m - 1 - row, c] = image[row, :, c]
    return output

def color_seam(seam_idx):
    output = np.copy(out_image)
    m, n = output.shape[: 2]
    for row in range(m):
        col = seam_idx[row]
        output[row][col] = 255
    return output

def seams_removal(num_pixel, rotation=False):
    print("seam removal...")
    global img_counter
    for i in range(num_pixel):
        energy_map = calc_energy_map()
        cumulative_map = cumulative_map_forward(energy_map)
        seam_idx = find_seam(cumulative_map)
        if(rotation):
            save_result(rotate_image(color_seam(seam_idx), 0).astype('uint8'), "out/"+str(img_counter)+".jpg")
            img_counter+=1
            # video.write(cv2.copyMakeBorder(rotate_image(out_image, 0).astype('uint8'), 0, i+1, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255]))
        else:
            save_result(color_seam(seam_idx).astype('uint8'), "out/"+str(img_counter)+".jpg")
            img_counter+=1
            # video.write(cv2.copyMakeBorder(out_image.astype('uint8'), 0, 0, 0, i + 1, cv2.BORDER_CONSTANT, value=[255,255,255]))
        delete_seam(seam_idx)

def seams_carving(delta_row, delta_col):
    
    global out_image
    print("seam carving...")
    
    # h,w,_ = out_image.shape
    # video = cv2.VideoWriter("out.avi", cv2.cv.CV_FOURCC(*'MJPG'), 15, (w,h))
    # remove column
    
    energy_map = calc_energy_map()
    plt.imshow(energy_map, 'gray')
    plt.show()
    
    if delta_col < 0:
        seams_removal(delta_col * -1)
    # insert column
    elif delta_col > 0:
        seams_insertion(delta_col)

    # remove row
    if delta_row < 0:
        out_image = rotate_image(out_image, 1)
        seams_removal(delta_row * -1, True)
        out_image = rotate_image(out_image, 0)
    # insert row
    elif delta_row > 0:
        out_image = rotate_image(out_image, 1)
        seams_insertion(delta_row, True)
        out_image = rotate_image(out_image, 0)

def main(input_file, out_height, out_width):
    print("initializing")
    
    global out_image

    # read in image and store as np.float64 format
    in_image = cv2.imread(input_file).astype(np.float64)
    in_height, in_width = in_image.shape[: -1]
    
    out_image = np.copy(in_image)

    print("starting...")
    delta_row, delta_col = int(out_height - in_height), int(out_width - in_width)
    seams_carving(delta_row, delta_col)


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Seam Carving")
    parser.add_argument('input_file', help="Input image")
    parser.add_argument('output_file', help="Output image")
    parser.add_argument('height', help="Target height")
    parser.add_argument('width', help="Target width")

    args = parser.parse_args()
    main(args.input_file, int(args.height), int(args.width))
    save_result(out_image, args.output_file)
