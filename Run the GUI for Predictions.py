import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

class SkinCancerDetectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("DermaScan AI - Skin Cancer Detection")
        self.master.geometry("1000x800")
        self.master.minsize(800, 600)
        self.configure_styles()
        
        # Initialize model and variables
        self.model = None
        self.image_path = None
        self.setup_model()
        
        # Create GUI components
        self.create_widgets()
        
    def configure_styles(self):
        """Configure custom styles for widgets"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color scheme
        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#3498DB',
            'success': '#27AE60',
            'danger': '#E74C3C',
            'background': '#ECF0F1'
        }
        
        # Configure styles
        self.style.configure('TFrame', background=self.colors['background'])
        self.style.configure('TLabel', background=self.colors['background'], 
                            font=('Helvetica', 12))
        self.style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'),
                            foreground=self.colors['primary'])
        self.style.configure('TButton', font=('Helvetica', 12, 'bold'),
                            padding=10, borderwidth=0)
        self.style.configure('Primary.TButton', foreground='white',
                            background=self.colors['secondary'])
        self.style.configure('Success.TLabel', foreground=self.colors['success'],
                            font=('Helvetica', 14, 'bold'))
        self.style.configure('Danger.TLabel', foreground=self.colors['danger'],
                            font=('Helvetica', 14, 'bold'))
    
    def setup_model(self):
        """Load the trained model with error handling"""
        try:
            self.model = tf.keras.models.load_model('skin_cancer_detection.h5')
        except Exception as e:
            messagebox.showerror("Model Error", 
                f"Failed to load model:\n{str(e)}\nPlease check model file.")
            self.master.destroy()
    
    def create_widgets(self):
        """Create and arrange GUI components"""
        # Main container
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        ttk.Label(header_frame, text="DermaScan AI", style='Header.TLabel'
                 ).pack(side=tk.LEFT)
        
        # Image display
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        self.canvas = tk.Canvas(self.image_frame, bg='white', 
                               highlightthickness=1, highlightbackground="#BDC3C7")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=20)
        
        self.btn_upload = ttk.Button(control_frame, text="Upload Image", 
                                    style='Primary.TButton', command=self.upload_image)
        self.btn_upload.pack(side=tk.LEFT, padx=10)
        
        self.btn_classify = ttk.Button(control_frame, text="Analyze", 
                                      style='Primary.TButton', state=tk.DISABLED,
                                      command=self.analyze_image)
        self.btn_classify.pack(side=tk.LEFT, padx=10)
        
        # Results panel
        self.result_frame = ttk.Frame(main_frame)
        self.result_frame.pack(fill=tk.X, pady=20)
        
        self.lbl_result = ttk.Label(self.result_frame, text="", 
                                   font=('Helvetica', 16, 'bold'))
        self.lbl_result.pack()
        
        self.lbl_confidence = ttk.Label(self.result_frame, text="")
        self.lbl_confidence.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        
        # Status bar
        self.status_bar = ttk.Label(self.master, text="Ready", 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """Handle image upload"""
        file_types = [('Image files', '*.jpg *.jpeg *.png')]
        path = filedialog.askopenfilename(filetypes=file_types)
        if path:
            self.image_path = path
            self.show_image(path)
            self.btn_classify.config(state=tk.NORMAL)
            self.clear_results()
            self.update_status("Image uploaded successfully")
    
    def show_image(self, path):
        """Display the uploaded image"""
        try:
            self.canvas.delete("all")
            img = Image.open(path)
            img.thumbnail((600, 600), Image.Resampling.LANCZOS)
            
            # Center the image on canvas
            x = (self.canvas.winfo_width() - img.width) // 2
            y = (self.canvas.winfo_height() - img.height) // 2
            
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")
    
    def analyze_image(self):
        """Process and classify the image"""
        if not self.image_path:
            return
            
        self.progress.start()
        self.update_status("Analyzing image...")
        self.master.update_idletasks()
        
        try:
            # Preprocess image
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("Failed to read image file")
            
            img = cv2.resize(img, (175, 175))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img)[0][0]
            self.display_results(prediction)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error processing image:\n{str(e)}")
            self.update_status("Error occurred during analysis")
        
        finally:
            self.progress.stop()
            self.update_status("Analysis complete")
    
    def display_results(self, prediction):
        """Show classification results with styling"""
        is_malignant = prediction >= 0.5
        confidence = prediction if is_malignant else 1 - prediction
        confidence_percent = f"{confidence * 100:.2f}%"
        
        result_text = "Malignant (Cancerous)" if is_malignant else "Benign (Non-Cancerous or First Stage)"
        color_style = 'Danger.TLabel' if is_malignant else 'Success.TLabel'
        
        self.lbl_result.config(text=result_text, style=color_style)
        self.lbl_confidence.config(text=f"Confidence: {confidence_percent}")
        
        # Update status with recommendation
        recommendation = "Consult a dermatologist immediately" if is_malignant \
                        else "Regular monitoring recommended"
        self.update_status(f"Result: {result_text} | {recommendation}")
    
    def clear_results(self):
        """Reset result display"""
        self.lbl_result.config(text="")
        self.lbl_confidence.config(text="")
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = SkinCancerDetectorGUI(root)
    root.mainloop()