import datetime
import logging
import time
import os
import re
import sys
import glob
import cv2
import numpy as np
import tkinter.filedialog as filedialog
import tkinter as tk
from tkinter import messagebox
from tkinter import font
from threading import Thread
from skimage.metrics import structural_similarity

def nothing(x):
    pass

# Create MainUI class.
class MainUI():
    """
    Class that serves as a the frontend for all of the programs user interactable functions.
    """
    def __init__(self) -> None:
        # Create class variables and objects.
        self.logger = logging.getLogger(__name__)
        self.window_is_open = True
        self.grid_size = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.font = "antiqueolive"

        # Image vars.
        self.images = []
        self.image_names = []
        self.list_view = None
        self.list_yscrollbar = None
        self.list_label = None

    def initialize_window(self) -> None:
        """
        Creates and populates all MainUI windows and components.

        Parameters:
        -----------
            None

        Returns:
        --------
            Nothing
        """
        self.logger.info("Initializing main window...")

        # Create UI component variables.
        self.window = None
        # Create new window object.
        self.window = tk.Tk()
        # Set window closing actions.
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        # Set window title.
        self.window.title("ImageTools")
        # Set window to the front of others.
        self.window.attributes("-topmost", True)
        self.window.update()
        self.window.attributes("-topmost", False)

        # Setup window grid layout.
        self.window.rowconfigure(self.grid_size, weight=1, minsize=50)
        self.window.columnconfigure(self.grid_size, weight=1, minsize=75)

        #######################################################################
        #               Create window components.
        #######################################################################
        # Create frame for title block.
        title_frame = tk.Frame(master=self.window, relief=tk.GROOVE, borderwidth=3)
        title_frame.grid(row=0, columnspan=10, sticky=tk.NSEW)
        title_frame.columnconfigure(0, weight=1)
        # Create frame for main image operations section.
        main_frame = tk.Frame(master=self.window, relief=tk.GROOVE, borderwidth=3)
        main_frame.grid(row=1, rowspan=9, column=0, columnspan=10, sticky=tk.NSEW)
        main_frame.rowconfigure(self.grid_size, weight=1)
        main_frame.columnconfigure(self.grid_size, weight=1)
        
        # Populate title frame.
        greeting = tk.Label(master=title_frame, text="Welcome to ImageTools UI!", font=(self.font, 18))
        greeting.grid()
        # Populate main frame.
        open_dir_deploy = tk.Button(master=main_frame, text="Open Image Directory", foreground="blue", background="white", command=self.ask_directory_callback)
        open_dir_deploy.grid(row=0, rowspan=1, column=0, columnspan=10, sticky=tk.NSEW)
        # Sub frame for image operations.
        image_operations_frame = tk.Frame(master=main_frame, relief=tk.GROOVE, borderwidth=2)   # Create frame from grouping vlan options.
        image_operations_frame.grid(row=1, rowspan=9, column=0, columnspan=2, sticky=tk.NSEW)
        image_operations_frame.rowconfigure(self.grid_size, weight=1)
        image_operations_frame.columnconfigure(self.grid_size, weight=1)
        img_difference = tk.Button(master=image_operations_frame, text="Find Difference", foreground="black", background="white", command=self.find_difference_callback)
        img_difference.grid(row=0, rowspan=1, column=0, columnspan=10, sticky=tk.NSEW)
        img_fade = tk.Button(master=image_operations_frame, text="Fade", foreground="black", background="white", command=self.img_fade_callback)
        img_fade.grid(row=1, rowspan=1, column=0, columnspan=10, sticky=tk.NSEW)
        # Subframe for selecting images.
        self.image_selection_frame = tk.Frame(master=main_frame, relief=tk.GROOVE, borderwidth=2)   # Create frame from grouping vlan options.
        self.image_selection_frame.grid(row=1, rowspan=9, column=2, columnspan=8, sticky=tk.NSEW)
        self.image_selection_frame.rowconfigure(self.grid_size, weight=1)
        self.image_selection_frame.columnconfigure(self.grid_size, weight=1)

    def ask_directory_callback(self) -> None:
        """
        This method is called when the Open Image Directory button is pressed. Runs after open image directory button has been pressed.

        Parameters:
        -----------
            None

        Returns:
        --------
            Nothing
        """
        # Clear images and image names lists.
        self.images.clear()
        self.image_names.clear()
        # Destroy old list view if already created.
        if self.list_view is not None:
            self.list_view.destroy()
            self.list_label.destroy()
            self.list_yscrollbar.destroy()

        # Get image directory.
        self.window.directory = filedialog.askdirectory()

        # Print log.
        self.logger.info(f"Searching {self.window.directory} for jpg, jpeg, and png images.")

        # Get jpg images.
        for filename in glob.glob(self.window.directory + "/*.jpg"):
            im = cv2.imread(filename)
            # Store image.
            self.images.append(im)
            # Store image name.
            self.image_names.append(filename.split('\\')[1])
        # Get jpeg images.
        for filename in glob.glob(self.window.directory + "/*.jpeg"):
            im = cv2.imread(filename)
            # Store image.
            self.images.append(im)
            # Store image name.
            self.image_names.append(filename.split('\\')[1])
        # Get png images.
        for filename in glob.glob(self.window.directory + "/*.png"):
            im = cv2.imread(filename)
            # Store image.
            self.images.append(im)
            # Store image name.
            self.image_names.append(filename.split('\\')[1])
        # Get jfif images.
        for filename in glob.glob(self.window.directory + "/*.jfif"):
            im = cv2.imread(filename)
            # Store image.
            self.images.append(im)
            # Store image name.
            self.image_names.append(filename.split('\\')[1])

        #######################################################################
        #   Populate image selector frame and store image names.
        #######################################################################
        # Create vertical scrollbar
        self.list_yscrollbar = tk.Scrollbar(master=self.image_selection_frame)
        self.list_yscrollbar.pack(side = tk.RIGHT, fill = tk.Y)
        # Create labels.
        self.list_label = tk.Label(master=self.image_selection_frame,
                    text = "Select multiple images below :  ",
                    font = ("Times New Roman", 10), 
                    padx = 10, pady = 10)
        self.list_label.pack()

        # Create and configure list box.
        self.list_view = tk.Listbox(master=self.image_selection_frame, selectmode = "multiple", 
                    yscrollcommand = self.list_yscrollbar.set)
        # Widget expands horizontally and vertically by assigning both to fill option
        self.list_view.pack(padx = 10, pady = 10, expand = tk.YES, fill = "both")
        
        # Add images to list view.
        for i, name in enumerate(self.image_names):
            self.list_view.insert(tk.END, name)
            self.list_view.itemconfig(i, bg = "gray")
        
        # Attach listbox to vertical scrollbar
        self.list_yscrollbar.config(command = self.list_view.yview)

    def find_difference_callback(self) -> None:
        """
        This method is called when the Find Difference button is pressed. Runs algorithm to find difference between two photos.

        Parameters:
        -----------
            None

        Returns:
        --------
            Nothing
        """
        # Check if list_view is not None.
        if self.list_view is not None:
            # Get list selections.
            image_selections = self.list_view.curselection()

            # Make sure two image are selected.
            if len(image_selections) == 2:
                image1, image2 = self.images[image_selections[0]], self.images[image_selections[1]]
                # Make sure images are the same size.
                if image1.shape == image2.shape:
                    # Launch seperate thread for image  processing.
                    thread = Thread(target=self.image_diff, args=(image1, image2))
                    thread.start()
                    # Show info to user.
                    messagebox.showinfo("Notice", "Press Q to close all processed image windows.")
                else:
                    # Show info to user.
                    messagebox.showinfo("Notice", "Images must be the same size.")
            else:
                # Show info to user.
                messagebox.showinfo("Notice", "Must select 2 images.")
        else:
            # Show info to user.
            messagebox.showinfo("Notice", "Must open image directory.")

    def image_diff(self, before, after):
        """
        Uses the SSIM algorithm to compare two images and find their difference.

        Parameters:
        -----------
            before - first image.
            after - second image.

        Returns:
        --------
            Nothing.
        """
        # Convert images to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between the two images
        (score, diff) = structural_similarity(before_gray, after_gray, full=True)
        self.logger.info("Image Similarity: {:.4f}%".format(score * 100))

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1] 
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")
        diff_box = cv2.merge([diff, diff, diff])

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(before.shape, dtype='uint8')
        filled_after = after.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(mask, [c], 0, (255,255,255), -1)
                cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        cv2.imshow('before', before)
        cv2.imshow('after', after)
        cv2.imshow('diff', diff)
        cv2.imshow('diff_box', diff_box)
        cv2.imshow('mask', mask)
        cv2.imshow('filled after', filled_after)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def img_fade_callback(self):
        """
        This method is called when the Image Fade button is pressed. Start thread for image fading.

        Parameters:
        -----------
            None

        Returns:
        --------
            Nothing
        """
        # Check if list_view is not None.
        if self.list_view is not None:
            # Get list selections.
            image_selections = self.list_view.curselection()

            # Make sure two image are selected.
            if len(image_selections) >= 2:
                # Launch seperate thread for image processing.
                selected_images = [self.images[i] for i in image_selections]

                # Make sure images are the same size.
                img_dims = [img.shape for img in selected_images]
                if img_dims.count(img_dims[0]) == len(img_dims):
                    thread = Thread(target=self.image_fade, args=([selected_images]))
                    thread.start()
                    # Show info to user.
                    messagebox.showinfo("Notice", "Press Q to close all processed image windows.")
                else:
                    # Show info to user.
                    messagebox.showinfo("Notice", "Images must be the same size.")
            else:
                # Show info to user.
                messagebox.showinfo("Notice", "Must select 2 images.")
        else:
            # Show info to user.
            messagebox.showinfo("Notice", "Must open image directory.")

    def image_fade(self, images):
        """
        Opens an OpenCV window with a trackbar to fade through a given set of images.

        Parameters:
        -----------
            images - A list of images. Must be the same size.

        Returns:
        --------
            Nothing
        """
        # Create a seperate window named 'controls' for trackbar
        cv2.namedWindow('controls')
        # Create trackbar in 'controls' window with name 'r''
        cv2.createTrackbar('r', 'controls', 0, 100, nothing)

        # Check if window is still open.
        while cv2.getWindowProperty('controls', cv2.WND_PROP_VISIBLE) >= 1:
            # Get fade amount from trackbar.
            fade_amount = int(cv2.getTrackbarPos('r','controls'))
            # Don't hit 100 because math.
            if fade_amount > 99:
                fade_amount = 99

            # Math for fading and switching images.
            fade_percent = int(fade_amount % (100 / (len(images) - 1)))
            fade_percent = fade_percent / (100 / (len(images) - 1)) # Remap range to 0-1
            image_step = int(fade_amount / (100 / (len(images) - 1)))
            if image_step >= len(images) - 1:
                image_step = len(images) - 2

            # Get images.
            img1 = images[image_step]
            img2 = images[image_step + 1]

            # Apply fade.
            dst = cv2.addWeighted(img1, 1 - fade_percent, img2, fade_percent, -1)
            cv2.imshow('controls', dst)

            if (cv2.waitKey(25) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break 


    def update_window(self) -> None:
        """
        Update the windows UI components and values.

        Parameters:
        -----------
            None

        Returns:
        --------
            Nothing
        """
        # Call main window event loop.
        self.window.update()

    def close_window(self) -> None:
        """
        This method is called when the main window closes.
        """
        # Print info.
        self.logger.info("Main window exit action has been invoked. Performing closing actions.")

        # Set bool value.
        self.window_is_open = False

        # Close window.
        self.window.destroy()

    def get_is_window_open(self) -> bool:
        """
        Returns if the window is still open and running.

        Parameters:
        -----------
            None

        Returns:
        --------
            Nothing
        """
        return self.window_is_open