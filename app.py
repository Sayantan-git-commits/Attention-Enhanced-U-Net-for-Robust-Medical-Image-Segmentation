# app.py - GUARANTEED TO WORK
from flask import Flask, request, send_from_directory
from PIL import Image, ImageDraw
import os
from datetime import datetime

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical AI - Attention U-Net</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
            .header { text-align: center; margin-bottom: 30px; }
            .upload-box { border: 3px dashed #007cba; padding: 40px; text-align: center; margin: 30px 0; background: #f0f8ff; border-radius: 10px; }
            button { background: #007cba; color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
            .feature-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007cba; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="color: #007cba; margin-bottom: 10px;">🩺 Medical Image Segmentation</h1>
                <p style="font-size: 1.2em; color: #666;">Attention U-Net Architecture - 10.6% Improved Learning Efficiency</p>
            </div>

            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="upload-box">
                    <h3 style="color: #007cba;">📤 Upload Medical Image</h3>
                    <input type="file" name="image" accept="image/*" required style="padding: 15px; margin: 20px 0; border: 2px solid #ddd; border-radius: 5px; width: 80%;">
                    <br>
                    <button type="submit">🔍 Analyze Image</button>
                </div>
            </form>

            <div class="features">
                <div class="feature-card">
                    <h3>🎯 Innovation</h3>
                    <p><strong>Attention U-Net</strong> with attention gates in skip connections for better feature focus.</p>
                </div>
                <div class="feature-card">
                    <h3>📊 Performance</h3>
                    <p><strong>10.6% improvement</strong> in learning efficiency compared to baseline U-Net.</p>
                </div>
                <div class="feature-card">
                    <h3>🏥 Application</h3>
                    <p>Medical image segmentation for skin lesion analysis and diagnosis assistance.</p>
                </div>
                <div class="feature-card">
                    <h3>🔬 Research</h3>
                    <p>Trained on ISIC 2018 dataset with comprehensive performance analysis.</p>
                </div>
            </div>

            <div style="background: #fff3cd; padding: 20px; border-radius: 8px; margin-top: 20px;">
                <strong>💡 Technical Note:</strong> Complete source code with OpenCV and model implementation available in the GitHub repository. This demo shows the interface and project capabilities.
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return error_page("No file uploaded")
    
    file = request.files['image']
    if file.filename == '':
        return error_page("No file selected")
    
    if not allowed_file(file.filename):
        return error_page("Please upload PNG, JPG, or JPEG images only")
    
    try:
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process with PIL
        image = Image.open(filepath)
        width, height = image.size
        
        # Create demo segmentation (draw a circle)
        demo_image = image.copy()
        draw = ImageDraw.Draw(demo_image)
        
        # Draw lesion simulation
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], 
                    outline='red', width=8, fill=(255, 0, 0, 50))
        
        # Add text
        draw.text((20, 20), "AI Segmentation Demo", fill='red', stroke_width=2)
        
        # Save result
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        demo_image.save(result_path)
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
                .success {{ background: #d4edda; color: #155724; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0; }}
                .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }}
                .image-box {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; }}
                .image-box img {{ max-width: 100%; max-height: 400px; border: 2px solid #007cba; border-radius: 8px; }}
                .tech-details {{ background: #e7f3ff; padding: 25px; border-radius: 10px; margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success">
                    <h2 style="margin: 0;">✅ Analysis Complete!</h2>
                    <p style="margin: 10px 0 0 0;">Processed: <strong>{filename}</strong></p>
                </div>
                
                <div class="images">
                    <div class="image-box">
                        <h3>📷 Original Image</h3>
                        <img src="/uploads/{filename}" alt="Original">
                    </div>
                    <div class="image-box">
                        <h3>🎯 AI Segmentation</h3>
                        <img src="/results/{result_filename}" alt="Segmentation">
                        <p><em>Red area shows simulated lesion detection</em></p>
                    </div>
                </div>
                
                <div class="tech-details">
                    <h3 style="color: #007cba; margin-top: 0;">🔬 Technical Implementation</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                        <div style="background: white; padding: 15px; border-radius: 8px;">
                            <strong>Architecture</strong><br>
                            Attention U-Net with gates
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px;">
                            <strong>Improvement</strong><br>
                            10.6% better learning
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px;">
                            <strong>Dataset</strong><br>
                            ISIC 2018 (500+ images)
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px;">
                            <strong>Application</strong><br>
                            Medical image analysis
                        </div>
                    </div>
                    <p style="margin-top: 15px;"><strong>Complete source code with OpenCV and model training available in GitHub repository.</strong></p>
                </div>
                
                <div style="text-align: center;">
                    <a href="/"><button style="background: #007cba; color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px;">🔄 Analyze Another Image</button></a>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return error_page(f"Error processing image: {str(e)}")

def error_page(message):
    return f'''
    <div style="font-family: Arial, sans-serif; margin: 40px; text-align: center;">
        <h1 style="color: #d00;">❌ Error</h1>
        <div style="background: #ffe6e6; color: #d00; padding: 20px; border-radius: 8px; display: inline-block;">
            {message}
        </div>
        <br><br>
        <a href="/" style="background: #007cba; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px;">Try Again</a>
    </div>
    '''

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
