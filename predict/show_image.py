import numpy as np
import os
import matplotlib
import matplotlib.pyplot
import matplotlib.animation
import tsahelper as tsa

INPUT_FOLDER = 'image_file/'
# read one file, this only read .aps
# other file catch exception
# return read in image data
def read_data(infile):
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')

    #skip header 512 bytes, read from 512
    fid.seek(512)
    if extension == '.aps':
        # print(h['word_type'])
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        # print("nx: {}| ny: {}| nt: {}".format(nx, ny, nt))
        # print(h['data_scale_factor'])
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    else:
        pritn('Please input .aps file only')
    fid.close()
    return data



# display one image
def plot_image(path):
    matplotlib.rc('animation', html='html5')
    data = tsa.read_data(path)
    print(data.shape)
    fig = matplotlib.pyplot.figure(figsize = (8,8))
    ax = fig.add_subplot(111)
    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
        return [im]
    return matplotlib.animation.FuncAnimation(fig, animate, frames=range(0,data.shape[2]), interval=250, blit=True)


# unit_test check if this image is successfully read and display
def unit_test():
    anm = plot_image(INPUT_FOLDER + '826b3b5eb25ddd6f7d2aed1e531e69b9.aps')
    matplotlib.pyplot.show()


if __name__ == "__main__":
    unit_test()