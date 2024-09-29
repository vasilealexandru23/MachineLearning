import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf

def content_cost_func(content_ouput, generated_output):
    """
    Function that computes the content-cost function.
    J_content = 1 / (4 * n_h * n_w * n_c * norm2(content_output - generated_output) ** 2)

    Arguments:
    content_ouput -- forward propagation of the content image in some layer l
    generated_output -- forward propagation of the generated image in some layer l
    * Both have the shape (1, n_h, n_w, n_c)

    Returns:
    J_content -- computed scalar content cost.
    """
    a_C = content_ouput[-1]
    a_G = generated_output[-1]

    # Retrieve the shape of the tensors
    _, n_h, n_w, n_c = a_C.shape

    # Reshape a_C and a_G into matrices 
    a_C_unrolled = tf.reshape(a_C, shape=[1, n_h * n_w, n_c])
    a_G_unrolled = tf.reshape(a_G, shape=[1, n_h * n_w, n_c])

    # Compute cost
    J_content = 1 / (4 * n_h * n_w * n_c) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

def gram_matrix(A):
    """
    This function computes the correlation between the channels applied on
    a filter in a layer of the VGG. The convention is that the tensor
    is already reshaped into (n_C, n_H * n_W). G = A * A.T -> style matrix.

    Arguments:
    A -- filter tensor

    Returns:
    G -- gram/style matrix G = (n_C, n_C)
    """

    G = tf.matmul(A, tf.transpose(A))

    return G

def layer_style_cost_func(a_S, a_G):
    """
    This function computes the style cost function applied on forward propagation
    of the style and generated images in same layer l.
    J_style_cost = 1 / (2 * n_h * n_w * n_c) ** 2 * norm(G(a_S) - G(a_G) ** 2)

    Arguments:
    a_S -- tensor with dimensions (1, n_h, n_w, n_c) = hidden layer activation in layer l of image S
    a_G -- tensor with dimensions (1, n_h, n_w, n_c) = hidden layer activation in layer l of image G

    Returns
    J_style_layer = tensor representing a scalar value, style cost
    """

    # Retrieve dimensions
    _, n_h, n_w, n_c = a_S.get_shape().as_list()

    # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
    a_S = tf.reshape(tf.transpose(a_S), shape=[n_c, n_h * n_w])
    a_G = tf.reshape(tf.transpose(a_G), shape=[n_c, n_h * n_w])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = 1 / (2 * n_h * n_w * n_c) ** 2 * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
    return J_style_layer

# Add the lambda wegiht over layers
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    J_style(S, G) = (sum over l layers) lambda(l) * J_style_layer(S, G)
    
    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        # Compute style_cost for the current layer
        J_style_layer = layer_style_cost_func(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    J = alpha * J_content + beta * J_style

    return J

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Import pre-trained weights into model
image_size = 400
model = tf.keras.applications.VGG19(include_top=False,
                                    input_shape = (image_size, image_size, 3),
                                    weights='Pretrained-Model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Print the model architecture
print(model.summary())

# Get the content image
content = Image.open("./images/jim.jpg")
content = np.array(content.resize((image_size, image_size)))
content = tf.constant(np.reshape(content, ((1,) + content.shape)))

# Get the style image
style = Image.open("./images/style_gogh.jpg")
style = np.array(style.resize((image_size, image_size)))
style = tf.constant(np.reshape(style, ((1,) + style.shape)))

# Initialize the "generated" image as a noisy image created from the content image
generated_image = tf.Variable(tf.image.convert_image_dtype(content, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

# Define the content layer and build the model
content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layer_outputs(model, STYLE_LAYERS + content_layer)

content_target = vgg_model_outputs(content)  # Content encoder
style_targets = vgg_model_outputs(style)     # Style encoder

# Assign the content image to be the input of the VGG model.  
# Set a_C to be the hidden layer activation from the layer we have selected
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

# Assign the input of the model to be the "style" image 
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        
        # Compute a_G as the vgg_model_outputs for the current generated image
        a_G = vgg_model_outputs(generated_image)
        
        # Compute the style cost
        J_style = compute_style_cost(a_G, a_S)

        # Compute the content cost
        J_content = content_cost_func(a_G, a_C)
        # Compute the total cost
        J = total_cost(J_content, J_style, alpha = 10, beta = 40)
        
    # Compute the gradients of the total cost with respect to the generated image
    grads = tape.gradient(J, generated_image)

    # Update the generated image
    optimizer.apply_gradients([(grads, generated_image)])

    # Clip the generated image
    generated_image.assign(clip_0_1(generated_image))

    return J

# Assign the generated image to the variable generated_image
generated_image = tf.Variable(generated_image)

# Train the model
epochs = 1000
for i in range(epochs):
    print(f"EPOCH: {i}")
    train_step(generated_image)


# Show the 3 images in a row
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
imshow(content[0])
ax.title.set_text('Content image')
ax = fig.add_subplot(1, 3, 2)
imshow(style[0])
ax.title.set_text('Style image')
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()
