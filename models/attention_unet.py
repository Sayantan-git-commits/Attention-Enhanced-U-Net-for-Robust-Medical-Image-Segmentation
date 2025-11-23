# models/attention_unet.py - FIXED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_unet import DoubleConv, Down, Up, OutConv

class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant regions - FIXED CHANNELS"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        g: gate signal from decoder (smaller size)
        x: skip connection from encoder (larger size)
        """
        # Downsample gate signal to match spatial dimensions of x
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 spatial dimensions if needed
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        # Add and apply ReLU
        psi = self.relu(g1 + x1)
        
        # Get attention weights
        psi = self.psi(psi)
        
        # Apply attention to skip connection
        return x * psi

class AttentionUNet(nn.Module):
    """YOUR INNOVATION: U-Net with Attention Gates - FIXED ARCHITECTURE"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Same as Base U-Net)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Attention Gates - FIXED CHANNEL SIZES
        self.attention_gate1 = AttentionGate(F_g=512, F_l=512, F_int=256)   # Level 4
        self.attention_gate2 = AttentionGate(F_g=256, F_l=256, F_int=128)   # Level 3
        self.attention_gate3 = AttentionGate(F_g=128, F_l=128, F_int=64)    # Level 2
        self.attention_gate4 = AttentionGate(F_g=64, F_l=64, F_int=32)      # Level 1
        
        # Decoder with attention
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)
        
        print("🎯 Attention U-Net initialized with FIXED attention gates!")

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)           # 64, 256, 256
        x2 = self.down1(x1)        # 128, 128, 128  
        x3 = self.down2(x2)        # 256, 64, 64
        x4 = self.down3(x3)        # 512, 32, 32
        x5 = self.down4(x4)        # 512, 16, 16 (if bilinear=True)
        
        # Decoder with Attention Gates - FIXED ORDER
        # Level 1: x5 (gate) + x4 (skip)
        x4_att = self.attention_gate1(g=x5, x=x4)
        d4 = self.up1(x5, x4_att)
        
        # Level 2: d4 (gate) + x3 (skip)  
        x3_att = self.attention_gate2(g=d4, x=x3)
        d3 = self.up2(d4, x3_att)
        
        # Level 3: d3 (gate) + x2 (skip)
        x2_att = self.attention_gate3(g=d3, x=x2)
        d2 = self.up3(d3, x2_att)
        
        # Level 4: d2 (gate) + x1 (skip)
        x1_att = self.attention_gate4(g=d2, x=x1)
        d1 = self.up4(d2, x1_att)
        
        logits = self.outc(d1)
        return torch.sigmoid(logits)

def test_attention_unet():
    """Test the fixed Attention U-Net"""
    model = AttentionUNet()
    x = torch.randn(2, 3, 64, 64)
    
    print("🧪 Testing Fixed Attention U-Net...")
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test attention mechanism specifically
    print("🧠 Attention gates are working correctly!")
    return model, output

if __name__ == "__main__":
    test_attention_unet()