import torch
from torchvision import transforms
import PIL




def run_model(image):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
    #transformations   Scale the image to the 32 x 32 size neural network expects , convert it into a tensor and normalise
    transform = transforms.Compose([       transforms.Resize((32,32)),
                                           transforms.ToTensor(),
                                           normalize])

    model = torch.load('Final_model.tar')
    out = model(transform(image).view(1,3,32,32))[0].tolist()

    return class_names[out.index(max(out))]
if __name__ == '__main__':
    img = PIL.Image.open('2-dog.jpg')
    print(run_model(img))
