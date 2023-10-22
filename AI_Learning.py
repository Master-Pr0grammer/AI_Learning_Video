import torch, PIL, time, cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.random.manual_seed(0)

# ---------------------------------------- HYPER PARAMETERS ----------------------------------------

#image file, and other options
image_file = 'face.png'                     # filename of training image
real_time = True                            # makes video realtime
save_final = True                           # saves final model ouput
final_image_filename = 'final_output.png'   # filename of final model ouput

#Hyper parameters
hidden_neurons = 50
learning_rate = 0.05
num_epoch = 600


#choose device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f'Device Set To: {device}')


# ---------------------------------------- Generate Data ----------------------------------------

#load dataset
img = PIL.Image.open(image_file)
width, height = img.size
numpydata = np.asarray(img)
numpydata = numpydata.mean(axis=2).astype(np.uint8)

y = torch.tensor(numpydata, dtype=torch.float32)
y = y.view(height*width, 1)
y = y / 255 #flatten to fit between 0 and 1

# Create a grid of x and y coordinates
x_coords = torch.arange(1, width+1, dtype=torch.float32)
y_coords = torch.arange(1, height+1, dtype=torch.float32)

# Create coordinate grids using broadcasting
x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
x = torch.stack((x_grid, y_grid), dim=-1)
x = x.view((height*width, 2))

#move data to device
x = x.to(device)
y = y.to(device)


# ---------------------------------------- AI MODEL DEFINITION ----------------------------------------

# create AI model
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        
        #Layers initialization
        self.fc1 = nn.Linear(2, hidden_neurons)
        self.bn1 = nn.BatchNorm1d(hidden_neurons)

        self.fc2 = nn.Linear(hidden_neurons,hidden_neurons)
        self.bn2 = nn.BatchNorm1d(hidden_neurons)

        self.fc3 = nn.Linear(hidden_neurons,hidden_neurons)
        self.bn3 = nn.BatchNorm1d(hidden_neurons)

        self.fc4 = nn.Linear(hidden_neurons,1)
        self.bn4 = nn.BatchNorm1d(1)

        #Functions initialization
        self.sigmoid = nn.Sigmoid()
        self.Leaky_ReLU = nn.LeakyReLU()
        self.loss_func = nn.MSELoss()
        self.optimizer= torch.optim.AdamW(self.parameters(), lr=learning_rate)

        self.loss_data = []

    #run forward pass
    def forward(self, x, targets= None):
        #run through all layers
        x = self.fc1(x)
        x = self.Leaky_ReLU(self.bn1(x))

        x = self.fc2(x)
        x = self.Leaky_ReLU(self.bn2(x))

        x = self.fc3(x)
        x = self.Leaky_ReLU(self.bn3(x))

        x = self.fc4(x)
        output = self.sigmoid(self.bn4(x))

        if targets is None:
            return output
        else:
            loss = self.loss_func(output, targets)
            return output, loss
        

    # -------------------- Other Sampling Functions --------------------

    #create image object from output
    def output_to_image(self, model_output):
        model_output = model_output * 255 #rescale output from, 1-0 to 0-255
        pixel_tensor = model_output.view(width, height).cpu().detach().numpy().astype(np.uint8) #convert output to numpy array with correct shape
        img = PIL.Image.fromarray(pixel_tensor) #create image sample
        return img
    
    #create and save final video
    def create_video(self, image_list, output_filename, framerate):
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, framerate, (width, height))

        # Convert Pillow images to OpenCV format and write to video
        for img in image_list:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            out.write(img_cv)

        # Release the VideoWriter
        out.release()

# --------------- Training loop set up ---------------

#build model
model = Brain()
model.to(device)

#Training loop
image_list = []

#sample base image
model.eval()
output = model(x)
image_list.append(model.output_to_image(output))

# ---------------------------------------- MAIN TRAINING LOOP ----------------------------------------
model.train()
start=time.time()
for i in tqdm(range(num_epoch)):
    #run forward pass
    output, loss = model(x,y)
    model.loss_data.append(loss.cpu().detach().item())

    # Backward and optimize
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    #sample from training step and save to frames list
    image_list.append(model.output_to_image(output))

#get train time in order to calcualte real time frame rate 
train_time= time.time()-start


# ---------------------------------------- GET TRAINING DATA AND VIDEO ----------------------------------------
print('Final Loss:',model.loss_data[-1])

#plot training data
plt.plot(model.loss_data)
plt.title('Loss VS. Epoch')
plt.show()

if save_final:
    image_list[-1].save(final_image_filename)

output_filename = "output_video.mp4"
if real_time:
    framerate = int(len(image_list)/train_time)  # Adjust frame rate to real time
else:
    framerate = 60  # Adjust frame rate to be 60 steps/second (60 FPS)
model.create_video(image_list, output_filename, framerate)