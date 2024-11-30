#importing all the libraries

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os
import cv2
import random
import pywt
import math
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/outputs/'

# Ensure the output directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Algorithm-1: LSB Steganography

# Part-1: LSB Encoding
def lsb_encode(image, message):
    binary_message = ''.join(format(ord(i), '08b') for i in message) + '00000000'  # Null terminator
    data_index = 0   # initializing the data_index as 0
    image_data = np.array(image)    # Flattening the image data

    # Iterate over all pixel values in the image
    for values in image_data:
        for pixel in values:
            if data_index < len(binary_message):
                # Modifying lsb to encode the message bits
                pixel[0] = int(format(pixel[0], '08b')[:-1] + binary_message[data_index], 2)
                data_index += 1
            # Break the loop if entire message is coded
            if data_index >= len(binary_message):
                break
    # Conversion of nupy array back to image
    return Image.fromarray(image_data)

#Part-2: LSB Decoding
def lsb_decode(image):
    binary_message = ""   # initialize with empty message
    image_data = np.array(image) # Convert image to numpy array

    # Decoding
    for values in image_data:
        for pixel in values:
            binary_message += format(pixel[0], '08b')[-1]  # Extract the LSB

    # Convert binary message to string until the null terminator
    message = "".join([chr(int(binary_message[i:i + 8], 2)) for i in range(0, len(binary_message), 8)])
    return message.split('\0')[0]  # Extract message before null terminator



# Algorithm-2: Discrete Cosine Transform 

# Part-1: DCT encoding
def dct_encode(cover_image, mess):
    
    # Convert the PIL image to NumPy array
    image = np.array(cover_image)

    # Convert mess to bits and add a delimiter at the end
    mess_bits = ''.join([format(ord(char), '08b') for char in mess]) + '1111111111111110'
    mess_ind = 0

    # Check whether the image is RGB or grayscale
    if len(image.shape) == 3:  # RGB image
        channels = [np.copy(image[:, :, c]) for c in range(3)]
    else:  # Grayscale image
        channels = [np.copy(image)]

    # Standard quant matrix JPEG compression
    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],  [72, 92, 95, 98, 112, 100, 103, 99]])

    # Process each channel independenly
    for channel in channels:
        # Get te image dimensions
        height, width = channel.shape

        # Divide the image into 8x8 bckls and apply dct
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                if mess_ind >= len(mess_bits):
                    break

                # Get the 8x8 bckl from channel
                bckl = channel[i:i+8, j:j+8]
                if bckl.shape != (8, 8):
                    continue

                # Applying DCT to the bckl
                bckl_dct = cv2.dct(np.float32(bckl))

                # Quantizing the DCT coeffs
                bckl_dct_quant = np.round(bckl_dct / quant_matrix).astype(int)

                # Embedding the mess bit in a mid-freq DCT coeff
                dct_val = bckl_dct_quant[4, 4]
                dct_val = (dct_val & ~1) | int(mess_bits[mess_ind])
                bckl_dct_quant[4, 4] = dct_val
                mess_ind += 1

                # Dequantise DCT coeffs and apply inverse DCT
                bckl_dct_dequant = bckl_dct_quant * quant_matrix
                bckl_idct = cv2.idct(np.float32(bckl_dct_dequant))

                # Replace the original bckl with modified bckl
                channel[i:i+8, j:j+8] = np.clip(bckl_idct, 0, 255)

        if mess_ind >= len(mess_bits):
            break

    # Merge the channels back if RGB
    if len(image.shape) == 3:  # RGB image
        for c in range(3):
            image[:, :, c] = channels[c]
    else:  # Grayscale image
        image = channels[0]

    # Convert the encoded image back to a PIL Image
    encoded_image = Image.fromarray(image.astype(np.uint8))
    return encoded_image


# Part-2: DCT Decoding
def dct_decode(stego_image):
    # Convert the PIL image to a NumPy array
    image = np.array(stego_image)

    # Check if the image is RGB or grayscale
    if len(image.shape) == 3:  # RGB image
        channels = [image[:, :, c] for c in range(3)]
    else:  # Grayscale image
        channels = [image]

    bits = ''

    # Standard quantization matrix for JPEG compression
    quant_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Process each channel independently
    for channel in channels:
        # Get the channel dims
        height, width = channel.shape

        # Divide the image into 8x8 bckls and extract hidden bits from DCT coefficients
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                # Get the 8x8 bckl from the channel
                bckl = channel[i:i+8, j:j+8]
                if bckl.shape != (8, 8):
                    continue

                # Apply DCT to the blck
                bckl_dct = cv2.dct(np.float32(bckl))

                # Quantize the DCT coefficients
                bckl_dct_quant = np.round(bckl_dct / quant_matrix).astype(int)

                # Extract the bit from the (4,4) DCT coefficient
                bit = int(bckl_dct_quant[4, 4]) & 1
                bits += str(bit)

                # Stop if the delimiter is found
                if bits.endswith('1111111111111110'):
                    bits = bits[:-16]  # Remove the delimiter
                    break

            if bits.endswith('1111111111111110'):
                break

        if bits.endswith('1111111111111110'):
            break

    # Convert the bits back to a mess
    mess = ''.join([chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)])
    return mess


def fft_encode(image, message):
    import numpy as np
    from PIL import Image

    # Convert the message to bits and append a 16-bit delimiter
    message_bits = ''.join(format(ord(char), '08b') for char in message) + '1111111111111110'
    messIterator = 0
    message_length = len(message_bits)

    # Convert the image to a NumPy array
    image_data = np.array(image, dtype=np.float32)

    # Get the image dimensions
    height, width, channels = image_data.shape

    # Define the quantization matrix
    quanti_mat = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Process the image in 8x8 blocks
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if messIterator >= message_length:
                break

            for channel in range(channels):
                block = image_data[i:i + 8, j:j + 8, channel]

                if block.shape[0] < 8 or block.shape[1] < 8:
                    continue

                block_fft = np.fft.fft2(block)
                block_fft_quantized = np.round(block_fft / quanti_mat)

                if messIterator <= message_length - 1:
                    magnitude = int(np.abs(block_fft_quantized[4, 4]))  # Convert to integer
                    phase = np.angle(block_fft_quantized[4, 4])
                    magnitude = (magnitude & ~1) | int(message_bits[messIterator])
                    block_fft_quantized[4, 4] = magnitude * np.exp(1j * phase)
                    messIterator += 1

                blockInvFFT = np.fft.ifft2(block_fft_quantized * quanti_mat)
                image_data[i:i + 8, j:j + 8, channel] = np.clip(np.real(blockInvFFT), 0, 255)

    encoded_image = Image.fromarray(np.uint8(image_data))
    return encoded_image

def fft_decode(image):
    import numpy as np

    ImgDataFrame = np.array(image, dtype=np.float32)
    height, width, channels = ImgDataFrame.shape

    QUANT_ARRAY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    bits = ""

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            for channel in range(channels):
                block = ImgDataFrame[i:i + 8, j:j + 8, channel]

                if block.shape[0] < 8 or block.shape[1] < 8:
                    continue

                FFTblock = np.fft.fft2(block)
                FFTblockQUANT = np.round(FFTblock / QUANT_ARRAY)
                magnitude = int(np.abs(FFTblockQUANT[4, 4]))  # Convert to integer
                bits += str(magnitude & 1)

                if len(bits) >= 16 and bits[-16:] == '1111111111111110':
                    chars = [chr(int(bits[k:k + 8], 2)) for k in range(0, len(bits) - 16, 8)]
                    return ''.join(chars)

    chars = [chr(int(bits[k:k + 8], 2)) for k in range(0, len(bits), 8)]
    return ''.join(chars)


def histogramShift_encode(cover_image, message):

    # Convert message to binary and add a delimiter
    MessBits = ''.join([format(ord(char), '08b') for char in message]) + '1111111111111110' # MessBits stores bit converted form of text message
    mes_it_indo = 0
    meslen = len(MessBits) # length of message signal

    # Convert the PIL image to a NumPy array
    cover_image = np.array(cover_image) # NumPy data fram created

    # Handle RGB images by applying histogram shifting to each channel independently
    # We have 3 channels, R G B

    if len(cover_image.shape) == 3:
        for channel in range(3):  # Iterate through R, G, B channels
            channel_image = cover_image[:, :, channel]
            hist = cv2.calcHist([channel_image], [0], None, [256], [0, 256]).astype(int).flatten()  # Histogram calculation function 

            # Find peak and zero points in the histogram
            peak = np.argmax(hist) # peak point where maxima of histogarm of image occurs
            zero = np.argmin(hist) # zero point where minima of histogram of image occurs

            # Ensure that peak and zero points are different
            if zero == peak:
                raise ValueError("Some issue with histogram, cannot apply this technique :( ") # Handles some critical cases you know :(

            # Shift histogram values to create space for embedding
            # Create gap for embedding the message

            ShiftChannel = channel_image.copy()
            if zero > peak:
                for intensity in range(zero, peak, -1):
                    if intensity + 1 <= 255:
                        ShiftChannel[channel_image == intensity] = intensity + 1
            else:
                for intensity in range(zero, peak):
                    if intensity - 1 >= 0:
                        ShiftChannel[channel_image == intensity] = intensity - 1

            # Gap created at peak location because hamne shift waise kiya
            # Embed message bits at peak intensity in the shifted channel

            for i in range(ShiftChannel.size):
                row = i // ShiftChannel.shape[1]
                col = i % ShiftChannel.shape[1]

                # Embed only if we are at the peak intensity location
                if ShiftChannel[row, col] == peak and mes_it_indo < meslen:
                    if MessBits[mes_it_indo] == '1':
                        # Use the created gap to embed without exceeding uint8 bounds
                        ShiftChannel[row, col] = np.clip(ShiftChannel[row, col] + 1, 0, 255)
                    mes_it_indo += 1

                # Stop embedding if the entire message has been embedded
                if mes_it_indo >= meslen:
                    break

            # Replace the modified channel back into the image
            cover_image[:, :, channel] = ShiftChannel

            # Stop embedding if the entire message has been embedded
            if mes_it_indo >= meslen:
                break

    else:  # For grayscale images
        hist = cv2.calcHist([cover_image], [0], None, [256], [0, 256]).astype(int).flatten()

        # Find peak and zero points in the histogram
        peak = np.argmax(hist) # peak point jaisa ki ham upar dekh chuke hai
        zero = np.argmin(hist) # zero point where minima of histogram of image occurs, life me kbhi low feel nhi rehna B+ :)

        # Ensure that peak and zero points are different
        if zero == peak:
            raise ValueError("Zero point not found yaar :(") # Error throw

        # Shift histogram values to create space for embedding
        ImgShifted = cover_image.copy()
        if zero > peak:
            for intensity in range(zero, peak, -1):
                if intensity + 1 <= 255:
                    ImgShifted[cover_image == intensity] = intensity + 1
        else:
            for intensity in range(zero, peak):
                if intensity - 1 >= 0:
                    ImgShifted[cover_image == intensity] = intensity - 1
        
        # Shifting done dana done to create gap at peak location

        # Embed message bits at peak intensity in the shifted image
        for i in range(ImgShifted.size):
            row = i // ImgShifted.shape[1]
            col = i % ImgShifted.shape[1]

            # Embed only if we are at the peak intensity location
            if ImgShifted[row, col] == peak and mes_it_indo < meslen:
                if MessBits[mes_it_indo] == '1':
                    # Use the created gap to embed without exceeding uint8 bounds
                    ImgShifted[row, col] = np.clip(ImgShifted[row, col] + 1, 0, 255) # Clip functionality also added so that we remain in range (0,255))
                mes_it_indo += 1

            # Stop embedding if the entire message has been embedded
            if mes_it_indo >= meslen:
                break

        cover_image = ImgShifted

    return Image.fromarray(cover_image)  # Convert back to PIL image
                                         # Stego image generated, haa haa :) :) :)


def histogramShift_decode(Raaz_IMG):
    # Convert the PIL image to a NumPy array
    Raaz_IMG = np.array(Raaz_IMG)

    bits = '' # empty string declared 

    # Handle RGB images by extracting bits from each channel
    if len(Raaz_IMG.shape) == 3:
        for channel in range(3):  # Iterate through R, G, B channels
            channel_image = Raaz_IMG[:, :, channel]
            hist = cv2.calcHist([channel_image], [0], None, [256], [0, 256]).astype(int).flatten()
            peak = np.argmax(hist) # detects peak of the histogram of Raaz image, as you are detective here

            # Extract message bits from peak intensity pixels
            for i in range(channel_image.size):
                row = i // channel_image.shape[1]
                col = i % channel_image.shape[1]

                # Append '1' if pixel value matches peak + 1, otherwise '0' 
                # Rgus us exact inverse process of the encoding scheme
                if channel_image[row, col] == peak + 1:
                    bits += '1'
                elif channel_image[row, col] == peak:
                    bits += '0'

                # Check for the end-of-message delimiter
                # Stop the process if delimiter is encountered (Don't literally encounter the delimiter :) )
            
                if bits.endswith('1111111111111110'):
                    bits = bits[:-16]  # Remove the delimiter
                    break

            # Stop decoding if the entire message has been extracted
            if bits.endswith('1111111111111110'):
                break

    else:  # For grayscale images
        hist = cv2.calcHist([Raaz_IMG], [0], None, [256], [0, 256]).astype(int).flatten()
        peak = np.argmax(hist) # jaisa aap dekh chuke hai upar

        # Extract message bits from peak intensity pixels
        for i in range(Raaz_IMG.size):
            row = i // Raaz_IMG.shape[1]
            col = i % Raaz_IMG.shape[1]

            # Append '1' if pixel value matches peak + 1, otherwise '0'
            if Raaz_IMG[row, col] == peak + 1:
                bits += '1'
            elif Raaz_IMG[row, col] == peak:
                bits += '0'
            # Guys, as we know, same thing hui thi upar
            # No sense of talking again and again
            # Be happy :)

            # Check for the end-of-message delimiter
            if bits.endswith('1111111111111110'):
                bits = bits[:-16]  # Remove the delimiter
                break

    # Convert bits back to the original message
    extracted_message = ''.join([chr(int(bits[i:i + 8], 2)) for i in range(0, len(bits), 8)])
    return extracted_message  # Hurray, we got the value ...

# Quantization matrix for DWT coefficients
qunt_Mtrx = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]);

def dwt_encode(coverImgg, message):
    """
    Embeds the secret message into frequency domain of a cover image using Disc. Wavelet Trans.
    inputs:
        coverImgg: Input cover image as a PIL Image.
        message: Secret message string to be embedded.
    outputs:
        Encoded image as a PIL Image with the secret message embedded.
    """
    # Convert the message to binary and append a delimiter
    mssgBits = ''.join(format(ord(char), '08b') for char in message) + '1111111111111110';
    mesag_ind = 0  # Tracks the current bit position in the message
    len_Messg = len(mssgBits);

    # Convert the cover image to a Numpy array
    image = np.array(coverImgg, dtype=np.float32);
    height, width, channels = image.shape;
    encoded_Imgg = np.zeros_like(image);

    channel = 0  # Start embedding from the first channel
    while channel < channels:
        chnl_Datas = image[:, :, channel];

        # Performs DWT on the current channel
        coeffs = pywt.wavedec2(chnl_Datas, 'haar', level=1);
        cA, (cH, cV, cD) = coeffs;

        # the below code has Flattened and quantized the cH coefficients
        cH_flat = cH.flatten();
        q_MtrxResizd = np.tile(qunt_Mtrx, (cH.shape[0] // 8 + 1, cH.shape[1] // 8 + 1));
        q_MtrxResizd = q_MtrxResizd[:cH.shape[0], :cH.shape[1]];
        cH_quantized = np.round(cH_flat / q_MtrxResizd.flatten()).astype(int);

        i = 0  # Start to embed bits in quantized coefficients
        while i < len(cH_quantized) and mesag_ind < len_Messg:
            # Modify the LSB of the coefficient to embed the message bit
            cH_quantized[i] = (cH_quantized[i] & ~1) | int(mssgBits[mesag_ind]);
            mesag_ind += 1;
            i += 1;

        # Dequantize the coefficients
        cH_dequantized = cH_quantized * q_MtrxResizd.flatten();
        cH_dequantized = cH_dequantized.reshape(cH.shape);

        # Reconstruct the channel with modified coefficients
        modified_coeffs = (cA, (cH_dequantized, cV, cD));
        reconstructed_channel = pywt.waverec2(modified_coeffs, 'haar');
        reconstructed_channel = np.clip(reconstructed_channel, 0, 255);

        # it will Store the reconstructed channel in encoded image
        encoded_Imgg[:, :, channel] = reconstructed_channel[:height, :width];
        channel += 1; # Move to the next channel

    return Image.fromarray(encoded_Imgg.astype(np.uint8));


def dwt_decode(encodedImagg):
    """
    Decodes the secret message embedded in the wavelet domain of the encoded image.
    Arguments taken:
        encodedImagg: Input stego image as a PIL Image.
    Returned value:
        The extracted secret message as a string.
    """
    # Convert the stego image to a NumPy array
    image = np.array(encodedImagg, dtype=np.float32)
    hieght, widthh, chnnlz = image.shape
    bits = ""  # Store the extracted bits

    channel = 0  # Start decoding from the first channel
    while channel < chnnlz:
        chnllDATA = image[:, :, channel]

        # Perform DWT on the current channel
        coeffs = pywt.wavedec2(chnllDATA, 'haar', level=1)
        cA, (cH, cV, cD) = coeffs

        # it will Flatten and quantize the cH coefficients (which were obtained from dwt)
        cH_flat = cH.flatten()
        q_matrix_resized = np.tile(qunt_Mtrx, (cH.shape[0] // 8 + 1, cH.shape[1] // 8 + 1))
        q_matrix_resized = q_matrix_resized[:cH.shape[0], :cH.shape[1]]
        cH_quantized = np.round(cH_flat / q_matrix_resized.flatten()).astype(int)

        i = 0  # Start extracting bits
        while i < len(cH_quantized):
            # Extract the LSB from each quantized coefficient
            bits += '1' if (cH_quantized[i] & 1) == 1 else '0'

            # Stop if the delimiter is found
            if bits.endswith('1111111111111110'):
                bits = bits[:-16]  # Remove the delimiter
                break
            i += 1

        if bits.endswith('1111111111111110'):
            break  # Exit if the message is fully extracted
        channel += 1

    # Convert binary bits back to a string
    mmsg_extrctd = ''
    i = 0
    while i < len(bits):
        byte = bits[i:i + 8]
        mmsg_extrctd += chr(int(byte, 2))
        i += 8

    return mmsg_extrctd

def canny_EdgeDetection(imgg, threshold1, threshold2):
    """
    Implements the Canny edge detection algorithm.
    
    Parameters:
    - imgg: Grayscale input imgg (2D Numpy array)
    - threshold1: Lower threshold which is for edge classification
    - threshold2: Upper threshold value used for edge classification
    
    Returns:
    - edges: Binary imgg with edges marked as 255 (white) and others as 0 (black).
    """

    # 1. Noise Reduction using Gaussian Blur
    smothend_Imagi = cv2.GaussianBlur(imgg, (5, 5), 1.4);

    # 2. Gradient Calculation using Sobel filters
    Gx = cv2.Sobel(smothend_Imagi, cv2.CV_64F, 1, 0, ksize=3); # Gradient in X dirction
    Gy = cv2.Sobel(smothend_Imagi, cv2.CV_64F, 0, 1, ksize=3);  # Gradient in Y dirction

    # Compute gradient magnitude and direction
    mgnitude = np.sqrt(Gx ** 2 + Gy ** 2);
    dirction = np.rad2deg(np.arctan2(Gy, Gx)) % 180 ; # direction normalized to [0, 180]

    # 3. Non-Maximum Suppression
    Rows, Columns = mgnitude.shape;
    nms = np.zeros_like(mgnitude, dtype=np.uint8);

    for i in range(1, Rows - 1):  # Loop through Rows (excluding border pixels)
        for j in range(1, Columns - 1):  # Loop through columns
            angle = dirction[i, j];
            q = r = 255;

            # Determine neighboring pixels in the gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = mgnitude[i, j + 1], mgnitude[i, j - 1];
            elif 22.5 <= angle < 67.5:
                q, r = mgnitude[i + 1, j - 1], mgnitude[i - 1, j + 1];
            elif 67.5 <= angle < 112.5:
                q, r = mgnitude[i + 1, j], mgnitude[i - 1, j];
            elif 112.5 <= angle < 157.5:
                q, r = mgnitude[i - 1, j - 1], mgnitude[i + 1, j + 1];

            # Suppress non-maximum pixels
            nms[i, j] = mgnitude[i, j] if mgnitude[i, j] >= max(q, r) else 0

    # 4. Double Thresholding
    strong_edges = (nms >= threshold2).astype(np.uint8) * 255;
    WeakEdges = ((nms >= threshold1) & (nms < threshold2)).astype(np.uint8) * 255;

    # 5. Edge Tracking by Hysteresis
    edges = strong_edges.copy();
    for i in range(1, Rows - 1):  # Track weak edges connected to strong edges
        for j in range(1, Columns - 1):
            if WeakEdges[i, j] == 255:
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2] == 255):
                    edges[i, j] = 255;

    return edges;

def edge_based_encode(cvrImgg, mssage_Secrt):
    # Convert the PIL image to a NumPy array
    cvrImgg = cv2.cvtColor(np.array(cvrImgg), cv2.COLOR_RGB2BGR);

    # Load and convert the image to grayscale for edge detection
    gray_image = cv2.cvtColor(cvrImgg, cv2.COLOR_BGR2GRAY);
    
    # Perform Canny Edge Detection so that edges can be found out
    edges = canny_EdgeDetection(gray_image, threshold1=100, threshold2=200);
    
    # Find the indices of the edge pixels (non-zero)
    edge_pixels = np.argwhere(edges != 0);
    
    # Convert the message to binary
    binMassage = ''.join(format(ord(i), "08b") for i in mssage_Secrt) + '1111111111111110';  # End of message delimiter
    
    # Check if the message can be embedded in the number of edge pixels
    if len(binMassage) > len(edge_pixels):
        raise ValueError("Message is too large to be embedded in the edge regions.");
    
    # Embedding the message in the least significant bits of the edge pixels
    data_index = 0;
    for (x, y) in edge_pixels:
        if data_index < len(binMassage):
            # Modify the least significant bit of the blue channel
            cvrImgg[x, y, 0] = int(format(cvrImgg[x, y, 0], '08b')[:-1] + binMassage[data_index], 2);
            data_index += 1;
        else:
            break

    # Convert back to PIL image for saving
    return Image.fromarray(cv2.cvtColor(cvrImgg, cv2.COLOR_BGR2RGB));

def edge_based_decode(stegimagee):
    # Convert the PIL image to a NumPy array
    stegimagee = cv2.cvtColor(np.array(stegimagee), cv2.COLOR_RGB2BGR);

    # Load and convert image to grayscale for edge detection
    gray_image = cv2.cvtColor(stegimagee, cv2.COLOR_BGR2GRAY);
    
    # Perform Canny Edge Detection
    edges = canny_EdgeDetection(gray_image, threshold1=100, threshold2=200);
    
    # Find the indices of edge pixels (that r non-zero)
    edge_pixels = np.argwhere(edges != 0);
    
    # Extract the data from least signif. bits of the edge pixels
    binary_DATA = "";
    for (x, y) in edge_pixels:
        binary_DATA += format(stegimagee[x, y, 0], '08b')[-1];
    
    # Split 8 bits and convert binary data to string
    all_bytes = [binary_DATA[i:i+8] for i in range(0, len(binary_DATA), 8)];
    
    # Extract message until the delimiter '1111111111111110' is found
    extrctd_MESSG = "";
    for byte in all_bytes:
        if byte == '11111111':
            break;
        extrctd_MESSG += chr(int(byte, 2));
    
    return extrctd_MESSG;  # Return the extracted message

def random_encode(input_cover_IMG, mes):

    # Convert mes to binary
    # Here, we are doing binary conversion of input text message
    binary_mes = ''.join([format(ord(char), "08b") for char in mes]) + '1111111111111110'  # End of mes delimiter

    # A delimiter is added to detect end of the sentence/ message block
    
    # Convert the PIL Image to a numpy array
    input_cover_IMG_np = np.array(input_cover_IMG)
    height, width, _ = input_cover_IMG_np.shape
    total_pixels = height * width
    
    if len(binary_mes) > total_pixels:
        raise ValueError("mes is too large to be embedded in the image.") # Made error handling, can be ommited as well, sannu ki !!!
    
    # Randomize pixel order
    random.seed(42)  # Seed for regeneration of same random sequence every time
                     # Here, we fixed a value of Seed = 42, in both the encode and decode side
                     # This seed information makes our encoding process very secure
                     # Very rare chances that one may detect the random sequence
                     # For furthur security purposes, we may include cryptographic algorithms 

    pixel_indices = list(range(total_pixels))
    random.shuffle(pixel_indices)
    
    # Embed mes bits in blue channel of each pixel
    img_copy = input_cover_IMG_np.copy()
    data_index = 0
    for pixel_index in pixel_indices:
        if data_index < len(binary_mes):
            x = pixel_index // width
            y = pixel_index % width
            
            # Modify the least significant bit of the blue channel 
            # Updtd the LSB (Har baar wahi krte hai, less sensitive :) )

            img_copy[x, y, 0] = int(format(img_copy[x, y, 0], "08b")[:-1] + binary_mes[data_index], 2)
            data_index += 1
        else:
            break
    
    return Image.fromarray(img_copy)  # Convert back to PIL Image and return
                                      # task comp -> return ho jao apne raaste....

# Function to decode the message from the image
def random_decode(Raaz_image):

    # Convert the PIL Image to a numpy array
    # Make proper data frame for analysis of the stego_image in which message is decoded

    Raaz_image_np = np.array(Raaz_image)
    height, width, _ = Raaz_image_np.shape
    total_pixels = height * width

    # Randomize pixel order
   
    random.seed(42)  # Seed for reproducibility
    pixel_indices = list(range(total_pixels))
    random.shuffle(pixel_indices)
    
    # This is what we did earlier in encode stage
    # Seed choose kiye the 42 value ka
    # same yaha le liya
    # we know the random order
    # start searching and extract the mess reqd
    # Extract binary data from the least significant bits of the blue channel

    binary_data = ""
    for pixel_index in pixel_indices:
        x = pixel_index // width
        y = pixel_index % width
        
        # Get the least significant bit of the blue channel
        binary_data += format(Raaz_image_np[x, y, 0], "08b")[-1]
    
    # Convert binary data to string message until the delimiter is found
    extracted_message = ""
    for i in range(0, len(binary_data), 8):
        byte = binary_data[i:i + 8]
        if byte == '11111111':  # End of message delimiter  # Delimiter daal diya.... :)
            break
        extracted_message += chr(int(byte, 2))

    
    return extracted_message    # Got the secret message 
                                # yayyyyy :) :) :)
                                # Now, you may get a job in Army intelligence, hehe, placement ho gaya, ab kya, drop kardo....

def get_optimal_range(difference):
    ranges = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]
    for (l, u) in ranges:
        if l <= difference <= u:
            return l, u
    return 0, 255  # Default range if none found

def get_optimal_range(difference):
    ranges = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]
    for (l, u) in ranges:
        if l <= difference <= u:
            return l, u
    return 0, 255  # Default range if none found

#Algorithm-8: Pixel Value Diferencing

#Part-1: Encoding
def pvd_encode(image, secret_data):
    # Convert to  NumPy array
    img_np = np.array(image)

    # Convert to 3 color matrices: RED_c, GREEN_c, BLUE_c
    blue_c, green_c, red_c = cv2.split(img_np)

    # secret data ko convert kre into binary
    bin_data = ''.join(format(ord(char), '08b') for char in secret_data)
    ind = 0  # To track the position in secret data

    # Function to perform embedding on a single color matrix, this function is reused many times
    def process_color_matrix(matrix, max_t):
        nonlocal ind

        rows, cols = matrix.shape
        for i in range(0, rows - 1, 2): 
            for j in range(0, cols - 1, 2): 
                if ind >= len(bin_data):
                    return matrix  # Stop when sara data khatm ho jaye

                # Get two consecutive pixels
                p1, p2 = matrix[i, j], matrix[i, j + 1]

                # Calculate the difference value d
                d = abs(int(p1) - int(p2))

                # Find optimal range and calculate t (number of bits we can embed)
                l, u = get_optimal_range(d)  # Get range based on difference
                w = u - l  # Width of range
                t = math.floor(math.log2(w))  # Calculate how many no of bits can be embedded

                if t <= max_t: 
                    # Convert bits from secret data to to decimal
                    bits = bin_data[ind:ind + t]
                    ind += len(bits)

                    # Convert the bits into decimal
                    b = int(bits, 2)

                    # Calculate new difference d'
                    d_prime = l + b

                    # Calc new pixel values
                    if p1 > p2:
                        p1_new = p2 + d_prime
                        p2_new = p2
                    else:
                        p1_new = p1
                        p2_new = p1 + d_prime

                    # new pixel condition checking
                    p1_new = np.clip(p1_new, 0, 255)
                    p2_new = np.clip(p2_new, 0, 255)

                    # Embed the modified pixels into the matrix
                    matrix[i, j] = p1_new
                    matrix[i, j + 1] = p2_new

        return matrix

    # Process sare matrix
    # although this is not he most efficient way but it works
    red_c = process_color_matrix(red_c, max_t=5)  # red channel ke liye maximum t=5
    green_c = process_color_matrix(green_c, max_t=3)  # green channel ke liye maximum t=3
    blue_c = process_color_matrix(blue_c, max_t=7)  # blue channel ke liye maximum t=7

    # get stego image
    stego_image = cv2.merge([blue_c, green_c, red_c])
    return Image.fromarray(stego_image) 



def pvd_decode(stego_image):
    # Convert into a NumPy array, obvious step hr baar ki trh
    stego_img_np = np.array(stego_image)

    # if stego_img_np.ndim != 3 or stego_img_np.shape[2] != 3:
    #     raise ValueError("Stego image must be in RGB format with 3 channels.")

    # decoding steps in exactly opposite way
    # Split the image into color matrices
    blue_c, green_c, red_c = cv2.split(stego_img_np)

    secret_data_bits = []

    # Function to process a color matrix for data extraction, this function is encapsulated within the parent function
    # as it is rwuired to be used multiple times
    def extract_from_matrix(matrix, max_t):
        nonlocal secret_data_bits

        rows, cols = matrix.shape
        for i in range(0, rows - 1, 2):  # Prevent exceeding row bounds
            for j in range(0, cols - 1, 2):  # Prevent exceeding column bounds
                # Get two consecutive pixels
                p1, p2 = matrix[i, j], matrix[i, j + 1]

                # Calculate the difference value d
                d = abs(int(p1) - int(p2))

                # Find the optimal range and calculate t
                l, u = get_optimal_range(d)
                w = u - l
                t = math.floor(math.log2(w))

                if t <= max_t:  # Exct data if t is within limit
                    # Exct bits and add to secret data
                    d_prime = d - l
                    bits = format(d_prime, f'0{t}b')
                    secret_data_bits.extend(bits)

    # Extracteing data from each channel
    extract_from_matrix(red_c, max_t=5)
    extract_from_matrix(green_c, max_t=3)
    extract_from_matrix(blue_c, max_t=7)

    # Converting binary data back to the string format
    secret_data = []
    for i in range(0, len(secret_data_bits), 8):
        if len(secret_data_bits) >= i + 8: 
            byte = ''.join(secret_data_bits[i:i + 8])
            secret_data.append(chr(int(byte, 2)))

    return ''.join(secret_data) 

def resize_images(cover_img, stego_img):
    
    # Resize stego image to match cover image dimensions if necessary.

    if cover_img.shape != stego_img.shape:
        stego_img = cv2.resize(stego_img, (cover_img.shape[1], cover_img.shape[0]))
    return cover_img, stego_img

def calculate_psnr(cover_img, stego_img):
    
    # PSNR b/w input and output img
    
    mse = calculate_mse(cover_img, stego_img)
    if mse == 0:
        return float('inf') 
    max_pixel_value = 255.0 # assuming maximum no of levels to be 255 as in most of the course
    psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr_value

def calculate_mse(cover_img, stego_img):
    
    # Calculate the MSE between cover image and stego image.
    
    return np.mean((cover_img.astype("float") - stego_img.astype("float")) ** 2)

def calculate_ssim(cover_img, stego_img):
    
    #Calculate the SSIM between the i/p & o/p image
    #Supports both grayscale and RGB images.
   
    # Determine win_size based on image dimensions (smallest dimension divided by 2, ensuring it is odd)
    min_dim = min(cover_img.shape[0], cover_img.shape[1])
    win_size = min(7, min_dim // 2 * 2 + 1)  # Sets win_size to 7 or a smaller odd number if min_dim is smaller

    # Set channel_axis for multichannel images
    if cover_img.ndim == 2:  # Grayscale image
        return ssim(cover_img, stego_img, win_size=win_size)
    else:  # RGB image
        ssim_value = ssim(cover_img, stego_img, win_size=win_size, channel_axis=2)
        return ssim_value

def calculate_ncc(cover_img, stego_img):

    # Calculate the NCC (Normalized Cross-Correlation) between cover and stego images.
    
    cover_img = cover_img.astype("float")
    stego_img = stego_img.astype("float")
    cover_img_exp = np.mean(cover_img)
    stego_img_exp = np.mean(stego_img)
    nume = np.sum((cover_img - cover_img_exp) * (stego_img - stego_img_exp))
    denom = np.sqrt(np.sum((cover_img - cover_img_exp) ** 2) * np.sum((stego_img - stego_img_exp) ** 2))
    return nume / denom

def calculate_payload_capacity(cover_img, pixels_used_for_embedding):

    # Calculate the Payload Capacity of the steganographic technique as a percentage.
    
    total_pixels = cover_img.shape[0] * cover_img.shape[1]
    payload_capacity = (pixels_used_for_embedding / total_pixels) * 100
    return payload_capacity

def calculate_metrics(original_img, encoded_img):
    # Convert images to arrays
    original_array = np.array(original_img)
    encoded_array = np.array(encoded_img)

    # Resize if necessary
    original_array, encoded_array = resize_images(original_array, encoded_array)

    # Calculate metrics
    psnr = calculate_psnr(original_array, encoded_array)
    mse = calculate_mse(original_array, encoded_array)
    ssim = calculate_ssim(original_array, encoded_array)

    return mse,psnr, ssim

def generate_histogram(image, filename):
    #Generate and save the histogram of an image.
    plt.figure()
    plt.hist(np.array(image).ravel(), bins=256, color='gray', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    histogram_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.savefig(histogram_path)
    plt.close()
    return histogram_path

def save_histogram(image, filename):
    # Create histogram for the image
    plt.figure()
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        plt.hist(image[:, :, i].flatten(), bins=256, color=color, alpha=0.6, label=f'{color.upper()} Channel')
    plt.legend(loc='upper right')
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # Save the histogram
    histogram_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.savefig(histogram_path)
    plt.close()
    return filename


# routing for app

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/encode', methods=['GET', 'POST'])
def encode_route():
    if request.method == 'POST':
        file = request.files['file']
        message = request.form['message']
        algorithm = request.form['algorithm']
        image = Image.open(file.stream).convert('RGB')

        if algorithm == "LSB":
            encoded_image = lsb_encode(image, message)
        elif algorithm == "DCT":
            encoded_image = dct_encode(image, message)
        elif algorithm == "Edge Based Embedding":
            encoded_image = edge_based_encode(image, message)
        elif algorithm == "Histogram Shifting":
            encoded_image = histogramShift_encode(image, message)
        elif algorithm == "Random Pixel Embedding":
            encoded_image = random_encode(image, message)
        elif algorithm == "FFT":
            encoded_image = fft_encode(image,message)
        elif algorithm ==  "DWT":
            encoded_image = dwt_encode(image,message)
        elif algorithm == "PVD":
            encoded_image = pvd_encode(image,message)
        # Save original (cover) image
        cover_filename = "original_image.png"
        cover_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_filename)
        image.save(cover_path)

        # Save encoded image
        encoded_filename = "encoded_image.png"
        encoded_path = os.path.join(app.config['UPLOAD_FOLDER'], encoded_filename)
        encoded_image.save(encoded_path)

        # Calculate metrics
        mse_value, psnr_value, ssim_value = calculate_metrics(np.array(image), np.array(encoded_image))

        # Save histograms
        original_histogram = save_histogram(np.array(image), "original_histogram.png")
        encoded_histogram = save_histogram(np.array(encoded_image), "encoded_histogram.png")

        return render_template(
            'result.html',
            message=True,
            original_filename=cover_filename,
            encoded_filename=encoded_filename,
            mse=mse_value,
            psnr=psnr_value,
            ssim=ssim_value,
            original_histogram=original_histogram,
            encoded_histogram=encoded_histogram,
        )
    return render_template('encode.html')

@app.route('/decode', methods=['GET', 'POST'])
def decode_route():
    if request.method == 'POST':
        file = request.files['file']
        algorithm = request.form['algorithm']
        image = Image.open(file.stream)

        if algorithm == "LSB":
            extracted_message = lsb_decode(image)
        elif algorithm == "DCT":
            extracted_message = dct_decode(image)
        elif algorithm == "Edge Based Embedding":
            extracted_message = edge_based_decode(np.array(image))
        elif algorithm == "Histogram Shifting":
            extracted_message = histogramShift_decode(image)
        elif algorithm == "Random Pixel Embedding":
            extracted_message = random_decode(image)
        elif algorithm == "FFT":
            extracted_message = fft_decode(image) 
        elif algorithm == "DWT":
            extracted_message = dwt_decode(image)   
        elif algorithm == "PVD":
            extracted_message = pvd_decode(image)               

        return render_template('result.html', extracted_message=extracted_message)

    return render_template('decode.html')

if __name__ == '__main__':
    app.run(debug=True)
