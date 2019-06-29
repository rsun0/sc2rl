
class GraphConvModel(nn.Module, Model):
    def __init__(self, nonspatial_act_size, spatial_act_size, device):
        # Calls nn.Module's constructor, probably
        super().__init__()
        self.nonspatial_act_size = nonspatial_act_size
        self.spatial_act_size = spatial_act_size
        self.device = device

        self.config = GraphConvConfigMinigames 
        self.spatial_width = self.config.spatial_width
        self.embed_size = 128
        self.fc1_size = 128
        self.fc2_size = 128
        self.fc3_size = 128
        self.action_size = nonspatial_act_size + 4
        self.hidden_size = 256
        self.action_fcsize = 256
        self.graph_lstm_out_size = self.hidden_size + self.fc3_size + self.action_size
        
        FILTERS1 = 256
        FILTERS2 = 128
        FILTERS3 = 64
        FILTERS4 = 32
        
        self.where_yes = torch.ones(1).to(self.device)
        self.where_no = torch.zeros(1).to(self.device)
        
        self.unit_embedding = nn.Linear(self.config.unit_vec_width, self.embed_size)
        self.W1 = nn.Linear(self.embed_size, self.fc1_size)
        self.W2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.W3 = nn.Linear(self.fc2_size, self.fc3_size)
        
        #self.fc1 = nn.Linear(self.hidden_size, self.action_fcsize)
        self.action_choice = nn.Linear(self.graph_lstm_out_size, nonspatial_act_size)
        
        self.value_layer = nn.Linear(self.graph_lstm_out_size, 1)
        
        self.tconv1 = torch.nn.ConvTranspose2d(self.graph_lstm_out_size, 
                                                    FILTERS1,
                                                    kernel_size=4,
                                                    padding=0,
                                                    stride=2)
                                                    
        self.tconv2 = torch.nn.ConvTranspose2d(FILTERS1,
                                                    FILTERS2,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        self.tconv3 = torch.nn.ConvTranspose2d(FILTERS2,
                                                    FILTERS3,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
                                                    
        self.tconv4 = torch.nn.ConvTranspose2d(FILTERS3,
                                                    FILTERS4,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
                                                    
        self.tconv5 = torch.nn.ConvTranspose2d(FILTERS4,
                                                    self.spatial_act_size,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        self.tconv1_bn = torch.nn.BatchNorm2d(FILTERS1)
        self.tconv2_bn = torch.nn.BatchNorm2d(FILTERS2)
        self.tconv3_bn = torch.nn.BatchNorm2d(FILTERS3)
        self.tconv4_bn = torch.nn.BatchNorm2d(FILTERS4)

        self.activation = nn.Tanh()
        
        self.LSTM_embed_in = nn.Linear(self.fc3_size+self.action_size, self.hidden_size)
        
        self.hidden_layer = nn.LSTM(input_size=self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        num_layers=1)
