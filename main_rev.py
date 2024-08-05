import io
import cv2
import numpy as np
from PIL import ImageTk, Image
from Crypto.Cipher import Salsa20
import hashlib
import struct
import base64
import math
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns 
from sewar.full_ref import mse
import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import ttk
from tkinter import filedialog


def encrypt(plaintext, secret):
    cipher = Salsa20.new(key=secret)
    msg = cipher.nonce + cipher.encrypt(plaintext)
    return msg

def decrypt(cipher, secret):
    msg_nonce = cipher[:8]
    ciphertext = cipher[8:]
    cipher = Salsa20.new(key=secret, nonce=msg_nonce)
    plaintext = cipher.decrypt(ciphertext)
    return plaintext

def plot_hist(hist, num_bins=256):
    # Calculate histogram values and bin edges
    values, bins, _ = plt.hist(
        hist, density=True, alpha=0, bins=num_bins
        )

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot histogram as a curved line
    plt.plot(bin_centers, values, '-')
    plt.title("Histogram")
    plt.show()

def calculate_entropy(hist):
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)
    return image_entropy

def get_hist_for_entropy(image_path, num_bins=256):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    hist, _ = np.histogram(gray_image.ravel(), bins=num_bins, range=(0, num_bins))

    return hist

def get_mse_psnr(ori_path, decrypt_path):
    ori = cv2.imread(ori_path)
    decr = cv2.imread(decrypt_path)

    if (ori.shape != decr.shape):
        return "",""

    err_mse = mse(ori, decr)
    err_psnr = cv2.PSNR(ori, decr)

    return err_mse, err_psnr


# Define UI class           
class ImageEncryptionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Encryption App")
        self.geometry("1150x650")
        
        self.image_path = tk.StringVar()
        self.encrypted_image_path = tk.StringVar()
        self.decrypted_image_path = tk.StringVar()
        self.img_width = tk.IntVar()
        self.img_height = tk.IntVar()
        self.secret_string = tk.StringVar()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Image path label and entry
        image_frame = tk.Frame(self)
        image_frame.pack()

        image_label = tk.Label(image_frame, text="Image Path:")
        image_label.pack(side=tk.LEFT, pady=20)

        image_entry = tk.Entry(image_frame, textvariable=self.image_path, state="readonly", width=50)
        image_entry.pack(side=tk.LEFT)

        open_button = ttk.Button(image_frame, text="Open", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=5)

        password_frame = tk.Frame(self)
        password_frame.pack()

        password_label = tk.Label(password_frame, text="Password: ")
        password_label.pack(side=tk.LEFT)

        password_entry = tk.Entry(password_frame, textvariable=self.secret_string, state="normal")
        password_entry.pack(side=tk.LEFT)

        # Encrypted and Decrypted image labels
        encrypted_frame = tk.Frame(self)
        encrypted_frame.pack()

        encrypted_label = tk.Label(self, text="Encrypted Image Path:")
        encrypted_label.pack()
        
        encrypted_entry = tk.Entry(self, textvariable=self.encrypted_image_path, state="readonly", width=50)
        encrypted_entry.pack()
        
        decrypted_label = tk.Label(self, text="Decrypted Image Path:")
        decrypted_label.pack()
        
        decrypted_entry = tk.Entry(self, textvariable=self.decrypted_image_path, state="readonly", width=50)
        decrypted_entry.pack()

        # Buttons
        button_frame = tk.Frame(self)
        button_frame.pack()

        encrypt_button = ttk.Button(button_frame, text="Encrypt", command=self.encrypt_image)
        encrypt_button.pack(side=tk.LEFT, padx=10, pady=20)

        decrypt_button = ttk.Button(button_frame, text="Decrypt", command=self.decrypt_image)
        decrypt_button.pack(side=tk.LEFT, padx=10)

        hist_ori_button = ttk.Button(button_frame, text="Original Image Histogram", command=self.show_original_histogram)
        hist_ori_button.pack(side=tk.LEFT, padx=10)

        hist_ori_button = ttk.Button(button_frame, text="Encrypted Image Histogram", command=self.show_encrypted_histogram)
        hist_ori_button.pack(side=tk.LEFT, padx=10)

        hist_enc_button = ttk.Button(button_frame, text="Decrypted Image Histogram", command=self.show_decrypted_histogram)
        hist_enc_button.pack(side=tk.LEFT, padx=10)
    
        labelframe = tk.Frame(self)
        labelframe.pack()

        # Labels for entropy and MSE/PSNR
        self.original_entropy_label = tk.Label(labelframe, text="Original Image Entropy:")
        self.original_entropy_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.encrypted_entropy_label = tk.Label(labelframe, text="Encrypted Image Entropy:")
        self.encrypted_entropy_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.decrypted_entropy_label = tk.Label(labelframe, text="Decrypted Image Entropy:")
        self.decrypted_entropy_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a new frame for the rest of the labels
        label2_frame = tk.Frame(self)
        label2_frame.pack()

        self.npcr_label = tk.Label(label2_frame, text="NPCR and UACI")
        self.npcr_label.pack(pady=10)

        self.mse_psnr_label = tk.Label(label2_frame, text="MSE and PSNR:")
        self.mse_psnr_label.pack(pady=10)

        # Image Display
        image_frame = tk.Frame(self)
        image_frame.pack()

        self.original_image_label = tk.Label(image_frame)
        self.original_image_label.grid(row=0, column=0, padx=10, pady=10)

        self.encrypted_image_label = tk.Label(image_frame)
        self.encrypted_image_label.grid(row=0, column=1, padx=10, pady=10)

        self.decrypted_image_label = tk.Label(image_frame)
        self.decrypted_image_label.grid(row=0, column=2, padx=10, pady=10)

    def show_original_histogram(self):
        if self.image_path.get():
            hist = get_hist_for_entropy(self.image_path.get())
            # print(self.image_path.get())
            plot_hist(hist)

    def show_encrypted_histogram(self):
        if self.encrypted_image_path.get():
            hist = get_hist_for_entropy(self.encrypted_image_path.get())
            # print(self.encrypted_image_path.get())
            plot_hist(hist)

    def show_decrypted_histogram(self):
        if self.decrypted_image_path.get():
            hist = get_hist_for_entropy(self.decrypted_image_path.get())
            # print(self.decrypted_image_path.get())
            plot_hist(hist)
    
    def update_entropy_labels(self, original_entropy, encrypted_entropy, decrypted_entropy):
        self.original_entropy_label.config(text="Original Image Entropy: " + str(original_entropy) + " bit/px")
        self.encrypted_entropy_label.config(text="Encrypted Image Entropy: " + str(encrypted_entropy) + " bit/px")
        self.decrypted_entropy_label.config(text="Decrypted Image Entropy: " + str(decrypted_entropy) + " bit/px")

    def update_mse_psnr_label(self, mse, psnr):
        self.mse_psnr_label.config(text="MSE: " + str(mse) + "  PSNR: " + str(psnr) + " db")
    
    def calc_npcr_uaci(self, img_ori_path, img_enc_path):
        # image_ori = Image.open(img_ori_path).convert("RGB")
        # image_enc = Image.open(img_enc_path).convert("RGB")

        image_ori = Image.open(img_ori_path).convert("L")
        image_enc = Image.open(img_enc_path).convert("L")

        array_ori = np.array(image_ori)
        array_enc = np.array(image_enc)

        total_pixels = array_ori.size
        differing_pixels = np.sum(array_ori != array_enc)
        npcr = differing_pixels / total_pixels

        # Inisialisasi variabel total
        total_diff = 0
        # Perulangan untuk setiap piksel pada citra
        for x in range(self.img_width):
            for y in range(self.img_height):
                # Ambil nilai intensitas piksel asli dan hasil pemrosesan
                original_pixel = image_ori.getpixel((x, y))
                processed_pixel = image_enc.getpixel((x, y))

                # Hitung selisih absolut antara intensitas piksel asli dan hasil pemrosesan
                diff = abs(original_pixel - processed_pixel)

                # Tambahkan selisih ke total
                total_diff += diff

        # Hitung UACI
        uaci = total_diff / (self.img_width*self.img_height)  # Skala nilai UACI ke rentang 0-1 (0-100%)

        self.npcr_label.config(text="NPCR: " + str(npcr * 100) + "%  UACI: " + str(uaci))

    def open_image(self):
        image_path = filedialog.askopenfilename(title='Buka Gambar')
        if image_path:
            try:
                img = Image.open(image_path)
                # get image size
                self.img_width, self.img_height = img.size
                # img = ImageTk.PhotoImage(img)
                self.image_path.set(image_path)
                
                # set encypted image to image_path
                self.encrypted_image_path.set(image_path)

                # Clear
                self.original_entropy_label.config(text="Original Image Entropy: " )
                self.encrypted_entropy_label.config(text="Encrypted Image Entropy: " )
                self.decrypted_entropy_label.config(text="Decrypted Image Entropy: " )
                self.npcr_label.config(text="NPCR:")
                self.mse_psnr_label.config(text="MSE and PSNR:")

            except Exception as e:
                mbox.showerror("Error", str(e))
                
            # Open the image
            original_img = Image.open(image_path)

            # Resize the image to 50x50
            # resized_img = original_img.resize((250, 250), Image.ANTIALIAS)
            resized_img = original_img.resize((250, 250), Image.Resampling.LANCZOS)

            # Convert the resized image to PhotoImage
            tk_img = ImageTk.PhotoImage(resized_img)

            # Configure the label to display the resized image
            self.original_image_label.config(image=tk_img)
            self.original_image_label.image = tk_img
        else:
            mbox.showinfo("Error", "Please provide an image path.")

    def getpath(file_path):
        a = path.split(r'/')
        # print(a)
        name = a[-1]
        return name

    def getfilename(file_path):
        a = path.split(r'/')
        fname = a[-1]
        a = fname.split('.')
        a = a[0]
        return a
        
    def encrypt_image(self):
        image_path = self.image_path.get()
        if image_path:
            try:
                # Read the image
                image_en = Image.open(image_path).convert("RGB")

                # Retrieve the pixel values
                pixel_data = list(image_en.getdata())

                # Extract the RGB channel values
                byte_data_en = bytearray()
                for pixel in pixel_data:
                    for value in pixel:
                        byte_data_en.append(value)
                
                # print('Byte baca image',len(byte_data_en))                
                secret_key = hashlib.sha256(self.secret_string.get().encode()).digest()
                cyphertext = encrypt(byte_data_en, secret_key)
                # print('Byte enkrip image',len(cyphertext))

                # Create a new image with PIL
                image_en = Image.new('RGB', (self.img_width, self.img_height))

                # Set the pixel values
                image_en.frombytes(cyphertext)
                # print(len(byte_data))

                # Save the encrypted image
                encrypted_image_path = filedialog.asksaveasfilename(defaultextension=".png")
                if encrypted_image_path:
                    image_en.save(encrypted_image_path)
                    self.encrypted_image_path.set(encrypted_image_path)
                    mbox.showinfo("Success", "Image encryption successful!")
                    # Display encrypted image
                    original_img = Image.open(encrypted_image_path)
                    # Resize the image to 50x50
                    resized_img = original_img.resize((250, 250), Image.Resampling.LANCZOS)
                    # Convert the resized image to PhotoImage
                    tk_img = ImageTk.PhotoImage(resized_img)
                    # Configure the label to display the resized image
                    self.encrypted_image_label.config(image=tk_img)
                    self.encrypted_image_label.image = tk_img

                else:
                    mbox.showinfo("Error", "Invalid save path.")
            except Exception as e:
                mbox.showerror("Error", str(e))
        else:
            mbox.showinfo("Error", "Please provide an image path.")
    
    def decrypt_image(self):
        encrypted_image_path = self.encrypted_image_path.get()
        if encrypted_image_path:
            try:
                # Open the encrypted image
                image_de = Image.open(encrypted_image_path).convert("RGB")

                # Retrieve the pixel values
                pixel_data_de = list(image_de.getdata())

                # Extract the RGB channel values
                byte_data_de = bytearray()
                for pixel in pixel_data_de:
                    for value in pixel:
                        byte_data_de.append(value)

                bytes_to_decrypt = byte_data_de + b'\0' * 8
                # print('baca enkrip dengan tambahan 8 byte:', len(bytes_to_decrypt))

                secret_key = hashlib.sha256(self.secret_string.get().encode()).digest()
                decrypted_bytes = decrypt(bytes_to_decrypt, secret_key)
                # print('plaintext:', len(decrypted_bytes))

                # Create a PIL Image object from the decrypted image data
                decrypted_image = Image.new('RGB', (self.img_width, self.img_height))

                decrypted_image.frombytes(decrypted_bytes)

                # Save the decrypted image
                decrypted_image_path = filedialog.asksaveasfilename(defaultextension=".png")
                if decrypted_image_path:
                    decrypted_image.save(decrypted_image_path)
                    self.decrypted_image_path.set(decrypted_image_path)
                    mbox.showinfo("Success", "Image decryption successful!")
                    # Display decrypted image
                    original_img = Image.open(decrypted_image_path)
                    # Resize the image to 50x50
                    resized_img = original_img.resize((250, 250), Image.Resampling.LANCZOS)
                    # Convert the resized image to PhotoImage
                    tk_img = ImageTk.PhotoImage(resized_img)
                    # Configure the label to display the resized image
                    self.decrypted_image_label.config(image=tk_img)
                    self.decrypted_image_label.image = tk_img

                    # Calculate histogram and entropy of the original image
                    ori_hist = get_hist_for_entropy(self.image_path.get())
                    ori_entropy = calculate_entropy(ori_hist)

                    # Calculate histogram and entropy of the encrypted image
                    encrypted_hist = get_hist_for_entropy(self.encrypted_image_path.get())
                    encrypted_entropy = calculate_entropy(encrypted_hist)

                    # Calculate histogram and entropy of the decrypted image
                    decrypted_hist = get_hist_for_entropy(self.decrypted_image_path.get())
                    decrypted_entropy = calculate_entropy(decrypted_hist)

                    if (self.image_path.get() != self.encrypted_image_path.get()):
                        # Update the entropy labels
                        self.update_entropy_labels(ori_entropy, encrypted_entropy, decrypted_entropy)

                        # Calculate npcr
                        self.calc_npcr_uaci(self.image_path.get(), self.encrypted_image_path.get())

                        # Calculate MSE and PSNR
                        mse, psnr = get_mse_psnr(self.image_path.get(), decrypted_image_path)

                        # Update the MSE/PSNR label
                        self.update_mse_psnr_label(mse, psnr)
                
                else:
                    mbox.showinfo("Error", "Invalid save path.")
            except Exception as e:
                mbox.showerror("Error", str(e))
        else:
            mbox.showinfo("Error", "Please provide the encrypted image path.")


# Create and run the UI
app = ImageEncryptionApp()
app.mainloop()