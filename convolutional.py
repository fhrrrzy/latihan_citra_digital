# %% [markdown]
# # TR3 Citra digital : Convolutional
# Fahruraji - 4203250014 - PSIK 20 B

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Menampilkan Citra Original

# %%
img = cv2.imread('lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

plt.imshow(img)

# %% [markdown]
# ### List kernel dalam file PPT

# %%
kernel = {
    "gaussian": [
        [0, -1, 0], 
        [-1, 4, -1], 
        [0, -1, 0]
    ],
    "unweighted": [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    "weighted": [
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]
    ],
    "sharper": [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    "intensified": [
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ]
}


# %% [markdown]
# ## Class Convolution
# 
# Terdapat beberapa method yaitu:
# - set_type(String) ->  "rgb" atau "gray"
# - set_kernel(array 3x3) ->  kernel
# - read_image(path) -> lokasi image
# - do_convolution()
# - do_convolution_gray()
# - do_convolution_rgb()
# - plotting()
# 
# 
# *semua method langsung set input ke attribut object*
# 
# ### set_type(string)
# lakukan set type sebelum membaca gambar dengan method read_image() hanya ada dua pilihan _"gray"_ atau _"rgb"_ \
# contoh : _set_type("rgb")_
# 
# ### set_kernel(array 3x3)
# untuk set kernel dalam objek dimana akan digunakan pada method do_convolution, untuk sementara hanya bisa menerima array 3x3 karna hitungannya masih untuk array 3x3 / static\
# contoh : _set_kernel([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])_
# 
# ### read_image(path)
# untuk membaca image berdasarkan path yang diberikan\
# contoh : _read_image("lenna.png")_
# 
# ### do_convolution()
# tanpa parameter, passing ke method selanjutnya berdasarkan type yang telah di set sebelumnya, jika rgb maka diteruskan ke proses *do_convolution_rgb()*, jika gray diteruskan ke proses *do_convolution_gray()* \
# contoh : do_convolution()
# 
# ### do_convolution_gray()
# melakukan perhitungan matrix 2d (gray) image dan matrix kernel
# 
# ### do_convolution_rgb()
# melakukan perhitungan matrix 3d (rgb) image dan matrix kernel
# 
# ### plotting()
# tanpa parameter, melakukan plotting atribut image result pada object
# contoh : _plotting()_

# %%
class Convolution:
    def __init__(self):
        kernel = None
        img = None
        type = None
        img_result = None

    def do_convolution_gray(self):
        self.kernel = np.array(self.kernel)
        sum_kernel = np.sum(self.kernel.ravel())
        img_placeholder = np.zeros([self.img.shape[0]-1, self.img.shape[1]-1], self.img.dtype)

        for i in range(1, self.img.shape[0]-2):
            for j in range(1, self.img.shape[1]-2):
                img_placeholder[i-1][j-1] = (
                    self.img[i-1][j-1] * self.kernel[0][0] + 
                    self.img[i-1][j]   * self.kernel[0][1] + 
                    self.img[i-1][j+1] * self.kernel[0][2] +

                    self.img[i][j-1]   * self.kernel[1][0] + 
                    self.img[i][j]     * self.kernel[1][1] + 
                    self.img[i][j+1]   * self.kernel[1][2] +

                    self.img[i+1][j-1] * self.kernel[2][0] + 
                    self.img[i+1][j]   * self.kernel[2][1] + 
                    self.img[i+1][j+1] * self.kernel[2][2] ) / (sum_kernel if sum_kernel > 1 else 1)

                img_placeholder[i-1][j-1] = 255 if img_placeholder[i-1][j-1] > 255 else 0 if img_placeholder[i-1][j-1] < 0 else img_placeholder[i-1][j-1]

        self.img_result = img_placeholder
        return self

    def do_convolution_rgb(self):
        self.kernel = np.array(self.kernel)
        sum_kernel = np.sum(self.kernel.ravel())
        img_placeholder = np.zeros([self.img.shape[0]-1, self.img.shape[1]-1, self.img.shape[2]], self.img.dtype)

        for i in range(1, self.img.shape[0]-2):
            for j in range(1, self.img.shape[1]-2):
                for rgb in range(self.img.shape[2]):
                    res = (
                        self.img[i-1][j-1][rgb] * self.kernel[0][0] + 
                        self.img[i-1][j][rgb]   * self.kernel[0][1] + 
                        self.img[i-1][j+1][rgb] * self.kernel[0][2] +

                        self.img[i][j-1][rgb]   * self.kernel[1][0] + 
                        self.img[i][j][rgb]     * self.kernel[1][1] + 
                        self.img[i][j+1][rgb]   * self.kernel[1][2] +

                        self.img[i+1][j-1][rgb] * self.kernel[2][0] + 
                        self.img[i+1][j][rgb]   * self.kernel[2][1] + 
                        self.img[i+1][j+1][rgb] * self.kernel[2][2] ) / (sum_kernel if sum_kernel > 1 else 1)

                    img_placeholder[i-1][j-1][rgb] = 255 if res > 255 else 0 if res < 0 else res

        self.img_result = img_placeholder
        return self

    def do_convolution(self):
        if self.type == 'gray':
            self.do_convolution_gray()
        else:
            self.do_convolution_rgb()
        return self

    def plotting(self):
        plt.subplot(121)
        plt.title('original image')
        plt.imshow(self.img, cmap='gray') if self.type == 'gray' else plt.imshow(self.img)
        plt.subplot(122)
        plt.title('convolution image')
        plt.imshow(self.img_result, cmap='gray') if self.type == 'gray' else plt.imshow(self.img_result)
        return self

    def set_type(self, type):
        self.type = type
        return self

    def read_image(self, path):
        img = cv2.imread(path)
        if self.type == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.type == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img = img
        return self

    def set_kernel(self, kernel):
        self.kernel = kernel
        return self



# %% [markdown]
# ### membut object lalu memanggil semua method yang di perlukan, disini kernel gaussian

# %%
img = Convolution() 
img.set_type('rgb').read_image('lenna.png').set_kernel(kernel["gaussian"]).do_convolution().plotting()

# %%
img = Convolution() 
img.set_type('gray').read_image('lenna.png').set_kernel(kernel["gaussian"]).do_convolution().plotting()

# %% [markdown]
# ### contoh memasukkan kernel secara langsung

# %%
img = Convolution()
img.set_type('rgb').read_image('lenna.png').set_kernel([[1,2,1],[2,4,2],[1,2,1 ]]).do_convolution().plotting()

# %%
img = Convolution()
img.set_type('gray').read_image('lenna.png').set_kernel([[1,2,1],[2,4,2],[1,2,1 ]]).do_convolution().plotting()

# %% [markdown]
# ### unweighted 3x3 smoothing kernel

# %%
img = Convolution()
img.set_type('rgb').read_image('lenna.png').set_kernel(kernel['unweighted']).do_convolution().plotting()

# %%
img = Convolution()
img.set_type('gray').read_image('lenna.png').set_kernel(kernel['unweighted']).do_convolution().plotting()

# %% [markdown]
# ### weighted 3x3 smoothing kernel with gaussian blur

# %%
img = Convolution()
img.set_type('rgb').read_image('lenna.png').set_kernel(kernel["weighted"]).do_convolution().plotting()

# %%
img = Convolution()
img.set_type('gray').read_image('lenna.png').set_kernel(kernel["weighted"]).do_convolution().plotting()

# %% [markdown]
# ### kernel to make image sharper

# %%
img = Convolution()
img.set_type('rgb').read_image('lenna.png').set_kernel(kernel["sharper"]).do_convolution().plotting()

# %%
img = Convolution()
img.set_type('gray').read_image('lenna.png').set_kernel(kernel["sharper"]).do_convolution().plotting()

# %% [markdown]
# ### Kernel intensified sharper

# %%
img = Convolution()
img.set_type('rgb').read_image('lenna.png').set_kernel(kernel["intensified"]).do_convolution().plotting()

# %%
img = Convolution()
img.set_type('gray').read_image('lenna.png').set_kernel(kernel["intensified"]).do_convolution().plotting()


