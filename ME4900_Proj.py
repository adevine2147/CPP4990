import tkinter as tk
from tkinter import PhotoImage
#i have issues with pillow on my laptop
#from PIL import Image, ImageTk 
import math

# Function to calculate endurance limit based on inputs
def calculate_endurance_limit():
    material = material_var.get() #string
    material_units = material_units_var.get() #string
    material_process = material_process_var.get() #string
    surface_finish = surface_finish_var.get() #string
    diameter = diameter_var.get() #double
    stress_type = stress_type_var.get() #string
    reliability = reliability_var.get() #boolean value
    reliability_percentage = reliability_percentage_var.get() #percentage
    load_type = load_type_var.get()

    # Perform the endurance limit calculation here
   
    k_a = ka(surface_finish,material, material_units, material_process)
    k_b = kb(diameter,material_units)
    k_c = kc(stress_type)
    k_e = ke(reliability,reliability_percentage)
    s_e = se(material, material_units,material_process)
    endurance_limit = k_a*k_b*k_c*k_e*s_e
    print('se', endurance_limit)
    if material_units == "SI":
         result_label.config(text=f"Endurance Limit: {endurance_limit:.2f} MPa") 
    else:
         result_label.config(text=f"Endurance Limit: {endurance_limit:.2f} Ksi") 



    # Display the result
    #result_label.config(text=f"Endurance Limit: {endurance_limit} MPa")
def se(material, units, process):
    SI_values_CD = [["AISI 1006",330, 280],["AISI 1010",180, 300],["AISI 1040",590, 490]]
    SI_values_HR = [["AISI 1006",300, 170],["AISI 1010", 320, 370],["AISI 1040", 520, 290], ["AISI 1060", 680, 370] ]
    English_values_CD = [["AISI 1006",48, 41],["AISI 1010", 53, 44],["AISI 1040", 85, 71] ]
    English_values_HR = [["AISI 1006", 43, 24],["AISI 1010", 47, 26],["AISI 1040", 76,42], ["AISI 1060", 98, 54] ]

    t = 0
    y = 0
    if units == "SI" and process == "HR":
        for i in range(len(SI_values_HR)):
                if material == SI_values_HR[i][0]:
                    t = SI_values_HR[i][1]
                    print(t)
                    y = SI_values_HR[i][2]
                    break
    elif units == "SI" and process == "CD":
        for i in range(len(SI_values_CD)):
                if material == SI_values_CD[i][0]:
                    t = SI_values_CD[i][1]
                    y = SI_values_CD[i][2]
                    break
    elif units == "English" and process == "CD":
        for i in range(len(English_values_CD)):
                if material == English_values_CD[i][0]:
                    t = English_values_CD[i][1]
                    y = English_values_CD[i][2]
                    break
    elif units == "English" and process == "HR":
        for i in range(len(English_values_HR)):
                if material == English_values_HR[i][0]:
                    t = English_values_HR[i][1]
                    y = English_values_HR[i][2]
                    break
    
    s = 0
    if units == "SI":
         if t > 1400: #MPA
              s = 700 #MPA
         elif t <= 1400: #MPA
              s = 0.5*t
    if units == "English":
         if t > 200: #ksi
            s = 100 #ksi
         if t <= 200: #ksi
            s = 0.5*t
         
    return s
    


def ka(finish,material,units,process):
    #["material", HR tensile, HR yield, CD tensile, CD yield]/ 1060 only HR
    SI_values_CD = [["AISI 1006",330, 280],["AISI 1010",180, 300],["AISI 1040",590, 490]]
    SI_values_HR = [["AISI 1006",300, 170],["AISI 1010", 320, 370],["AISI 1040", 520, 290], ["AISI 1060", 680, 370] ]
    English_values_CD = [["AISI 1006",48, 41],["AISI 1010", 53, 44],["AISI 1040", 85, 71] ]
    English_values_HR = [["AISI 1006", 43, 24],["AISI 1010", 47, 26],["AISI 1040", 76,42], ["AISI 1060", 98, 54] ]
    
    t = 0
    y = 0
    if units == "SI" and process == "HR":
        for i in range(len(SI_values_HR)):
                if material == SI_values_HR[i][0]:
                    t = SI_values_HR[i][1]
                    y = SI_values_HR[i][2]
    elif units == "SI" and process == "CD":
        for i in range(len(SI_values_CD)):
                if material == SI_values_CD[i][0]:
                    t = SI_values_CD[i][1]
                    y = SI_values_CD[i][2]
    elif units == "English" and process == "CD":
        for i in range(len(English_values_CD)):
                if material == English_values_CD[i][0]:
                    t = English_values_CD[i][1]
                    y = English_values_CD[i][2]
    elif units == "English" and process == "HR":
        for i in range(len(English_values_HR)):
                if material == English_values_HR[i][0]:
                    t = English_values_HR[i][1]
                    y = English_values_HR[i][2]

    finish_list_English = [["Ground", 1.21, -0.067], ["Machined (CD)", 2.00, -0.217], ['Hot-rolled (HR)', 11.0, -0.650], ["As-forged", 12.7, -0.758]] #Ksi
    finish_list_SI = [["Ground", 1.38, -0.067], ["Machined (CD)", 3.04, -0.217], ['Hot-rolled (HR)', 38.6, -0.650], ["As-forged", 54.9, -0.758]] #MPa
    a = 0
    b = 0
    if units == "SI":
        for i in range(len(finish_list_SI)):
            if finish == finish_list_SI[i][0]:
                a = finish_list_SI[i][1]
                b = finish_list_SI[i][2]
                break
    elif units == "English":
        for i in range(len(finish_list_English)):
            if finish == finish_list_English[i][0]:
                a = finish_list_English[i][1]
                b = finish_list_English[i][2]
                break
    print('a',a)
    print('b',b)
    print('t',t)
    ka = (a) * (math.pow(t,b))
    print('ka',ka)
    return ka

def kb(d,type):
    b = 0
    if type == "SI":
        b = math.pow((d/7.62), -0.107)
    else:
        b = math.pow((d/0.3), -0.107)
    print('b',b) 
    return b

def kc(stress):
    c = 0
    print('stress', stress)
    if stress == "Bending":
        c = 1
    elif stress == "Tensile":
        c = 0.85
    else:
        c = 0.59
    print("c",c)
    return c

def ke(n, percent):
    e = 0
    if n == True:
        if percent == 50:
            e = 1
        elif percent == 90:
            e = 0.897
        elif percent == 95:
            e = 0.868
        elif percent == 99:
            e = 0.814
        elif percent == 99.9999:
            e = 0.620
        else:
            e = 1 #default value
        print("e", e)
        return e
    else:
        print("e",e)
        return 1 #no effect on equation
        



    






#can someone add the image on the side
'''def update_image():
    selected_image = image_var.get()
    if selected_image == "Image 1":
        #img_path = ImageTk.PhotoImage(Image.open("97.jpg"))
        img_path = "images.jpg"
    elif selected_image == "Image 2":
        img_path = "image2.png"
    else:
        img_path = "image3.png"

    img = Image.open(img_path)
    img = img.resize((200, 200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to prevent garbage collection'''

# Create the main application window
app = tk.Tk()
app.title("Fatigue Analysis GUI")

# Material selection
material_label = tk.Label(app, text="Material:")
material_label.pack()
materials = ["AISI 1006", "AISI 1010", "AISI 1040", "AISI 1060"]
material_var = tk.StringVar()
material_var.set(materials[0])
material_option_menu = tk.OptionMenu(app, material_var, *materials)
material_option_menu.pack()
material_process_var = tk.StringVar()
material_process_var.set("HR")
material_process_pick1 = tk.Radiobutton(app, text = "HR", variable = material_process_var, value = "HR")
material_process_pick2 = tk.Radiobutton(app, text = "CD", variable = material_process_var, value = "CD")
material_process_pick1.pack()
material_process_pick2.pack()

# Material Property Units
material_label2 = tk.Label(app, text="Material Property Units:")
material_label2.pack()
material_units_var = tk.StringVar()
material_units_var.set("SI")
material_units_pick1 = tk.Radiobutton(app, text = "SI", variable = material_units_var, value = "SI")
material_units_pick2 = tk.Radiobutton(app, text = "English", variable = material_units_var, value = "English")
material_units_pick1.pack()
material_units_pick2.pack()

# Surface Finish selection
surface_finish_label = tk.Label(app, text="Surface Finish:")
surface_finish_label.pack()
surface_finish_types = ["Ground", "Machined (CD)", "Hot-rolled (HR)", "As-forged"]
surface_finish_var = tk.StringVar()
surface_finish_var.set(surface_finish_types[0])
surface_finish_option_menu = tk.OptionMenu(app,surface_finish_var, *surface_finish_types)
surface_finish_option_menu.pack()

#input diameter of shaft
diameter_label = tk.Label(app, text="Input shaft diameter: \n (Use same units as the material and make sure to use mm or in)")
diameter_label.pack()
diameter_var = tk.DoubleVar()
diameter_entry = tk.Entry(app, textvariable=diameter_var)
diameter_entry.pack()


# Stress Type selection
stress_type_label = tk.Label(app, text="Stress Type:")
stress_type_label.pack()
stress_type_var = tk.StringVar()
stress_type_var.set("Tensile")
stress_type_radio1 = tk.Radiobutton(app, text="Tensile", variable=stress_type_var, value="Tensile")
stress_type_radio2 = tk.Radiobutton(app, text="Compressive", variable=stress_type_var, value="Compressive")
stress_type_radio3 = tk.Radiobutton(app, text="Bending", variable=stress_type_var, value="Bending")
stress_type_radio1.pack()
stress_type_radio2.pack()
stress_type_radio3.pack()

# Reliability and Reliability Percentage inputs
reliability_label = tk.Label(app, text="Reliability:")
reliability_label.pack()
reliability_var = tk.BooleanVar()
reliability_checkbox = tk.Checkbutton(app, text="Applicable", variable=reliability_var)
reliability_checkbox.pack()
reliability_percentage_label = tk.Label(app, text="Reliability Percentage:")
reliability_percentage_var = tk.DoubleVar()
reliability_percentage_entry = tk.Entry(app, textvariable=reliability_percentage_var)
reliability_percentage_label.pack()
reliability_percentage_entry.pack()

# Load Type selection
load_type_label = tk.Label(app, text="Load Type:")
load_type_label.pack()
load_types = ["Completely reversing, Simple Loads", "Fluctuating Simple Loads", "Combination of Loading Modes"]
load_type_var = tk.StringVar()
load_type_var.set(load_types[0])
load_type_option_menu = tk.OptionMenu(app, load_type_var, *load_types)
load_type_option_menu.pack()

# Calculate Button
calculate_button = tk.Button(app, text="Calculate Endurance Limit", command=calculate_endurance_limit)
calculate_button.pack()

# Result Label
result_label = tk.Label(app, text="")
result_label.pack()


#below is part of image button that needs to be fixed
'''
image_label = tk.Label(app)
image_label.pack()

image_var = tk.StringVar()
image_var.set("Image 1")
image_option_menu = tk.OptionMenu(app, image_var, "Image 1", "Image 2", "Image 3")
image_option_menu.pack()

# Update Image Button
update_image_button = tk.Button(app, text="Update Image", command=update_image)
update_image_button.pack()'''

app.mainloop()
