import os, cv2
import matplotlib.pyplot as plt

input_dir = '../Input/'
output_dir = '../Output/'

sample_step_size     =   8; # \delta
max_dim              =   400; # maximum dimension of the image

# do not change the following

dilation_width_1     =   max(round(7*max_dim/400),1); # \omega
dilation_width_2     =   max(round(9*max_dim/400),1); # \kappa
blur_std             =   round(9*max_dim/400); # \sigma
color_space          =   2; # RGB: 1; Lab: 2; LUV: 4
whitening            =   1; # do color whitening

print 'Ready...'

command = './BMS %s %s %d %d %d %f %d %d %d' % (input_dir,output_dir, \
    sample_step_size,dilation_width_1,dilation_width_2, \
    blur_std, color_space, whitening, max_dim)
print (command);

os.system(command);

orig = cv2.imread(input_dir + 'Image_95.bmp')
final= cv2.imread(output_dir + 'Image_95.png')

plt.figure(1)
plt.subplot(121)
plt.xticks([]), plt.yticks([]);
plt.title('Test image')
plt.imshow(orig)
plt.subplot(122)
plt.xticks([]), plt.yticks([]);
plt.title('Eye-fixation map')
plt.imshow(final)
plt.show()

cv2.waitKey(0)