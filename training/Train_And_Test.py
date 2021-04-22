import torch

import util

class Train_And_Test():
    def __init__(self, model, optimizer, loss, train_loader, test_loader, validate_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validate_loader = validate_loader
        self.optimizer = optimizer
        self.loss = loss
        self.device = util.get_device()

        config = util.get_config()
        self.sample_length = int(config['sample_length'])
        self.lower_pitch_limit = int(config['lower_pitch_limit'])
        self.upper_pitch_limit = int(config['upper_pitch_limit'])
        self.classes = [x for x in range(self.lower_pitch_limit, self.upper_pitch_limit)]
        
        self.current_epoch = 0
        
        
    def train(self, epoch):
        self.model.train()
        num_correct = 0
        training_loss = 0.0
        self.current_epoch = epoch
            
        for batch_idx, (data, target) in enumerate(self.train_loader):
            
            data = data.float()
            data = data.to(self.device)
            target = target.to(self.device) # Target Shape = batch_size. i.e floats of midis. Not one-hot.
            
            data = data.view(-1, data.shape[1], 7) #  batch_size X sample_length X num_features
            self.optimizer.zero_grad()
            output, hidden = self.model(data) 
            output = output[:, int(self.sample_length/2), :] # Output Shape = batch_size, 1, num pitches
            output = output.view(-1, len(self.classes))

            running_loss = self.loss(output, target)
            training_loss += running_loss.item()
            running_loss.backward()
            self.optimizer.step()
            
            prediction = output.argmax(dim=1)
            num_correct += (prediction == target).sum().item()
            
            if batch_idx % 10 == 0:
                    print('Train Epoch: {}, Batch id: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), running_loss.item()))

        # # To improve legibility only one loss norm is plotted for an epoch (Goodfellow - pg 276 diag 8.1)
        training_loss /= len(self.train_loader)
        return num_correct, training_loss

    def test(self):
        self.model.eval()
        num_correct = 0
        testing_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                
                data = data.float()
                data = data.to(self.device)
                target = target.to(self.device) 

                data = data.view(-1, data.shape[1], 7) 
                output, hidden = self.model(data) 
                output = output[:, int(self.sample_length/2), :]
                output = output.view(-1, len(self.classes))

                running_loss = self.loss(output, target)
                testing_loss += running_loss.item()

                prediction = output.argmax(dim=1)
                num_correct += (prediction == target).sum().item()

        # # To improve legibility only one loss norm is plotted for an epoch (Goodfellow - pg 276 diag 8.1)
        testing_loss /= len(self.test_loader)
        return num_correct, testing_loss

    def validate(self):
        self.model.eval()
        num_correct = 0
        validation_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.validate_loader):
                
                data = data.float()
                data = data.to(self.device)
                target = target.to(self.device) 

                data = data.view(-1, data.shape[1], 7) 
                output, hidden = self.model(data) 
                output = output[:, int(self.sample_length/2), :]
                output = output.view(-1, len(self.classes))

                running_loss = self.loss(output, target)
                validation_loss += running_loss.item()

                prediction = output.argmax(dim=1)
                num_correct += (prediction == target).sum().item()

        # # To improve legibility only one loss norm is plotted for an epoch (Goodfellow - pg 276 diag 8.1)
        validation_loss /= len(self.validate_loader)
        return num_correct, validation_loss


    def save_model(self, model_location):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, model_location) 
        print('Saved Model')


    def load_model(self, model_location):
        checkpoint = torch.load(model_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.current_epoch = checkpoint['epoch']
        return self.model
