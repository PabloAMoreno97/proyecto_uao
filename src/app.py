#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import datetime
import tkinter as tk
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING

import img2pdf
from PIL import Image, ImageTk

# Importamos las funciones depuradas
from models.integrator import predict_neumonia
from data.read_img import read_dicom_file


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        # Atributos de la clase
        self.filepath = None
        self.patient_id = tk.StringVar()
        self.result_label = ""
        self.probability = ""
        self.original_image_pil = None
        self.heatmap_pil = None
        self.reportID = 0

        # Configuración de la ventana
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # Fuente en negrita
        self.fonti = font.Font(weight="bold")

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        # LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica",
                              font=self.fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap",
                              font=self.fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:",
                              font=self.fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:",
                              font=self.fonti)
        self.lab5 = ttk.Label(self.root,
                              text=("SOFTWARE PARA EL APOYO AL DIAGNÓSTICO "
                                    "MÉDICO DE NEUMONÍA"),
                              font=self.fonti)
        self.lab6 = ttk.Label(self.root, text="Probabilidad:",
                              font=self.fonti)

        # Etiquetas para mostrar la imagen y el heatmap
        self.image_label_frame = ttk.Frame(self.root,
                                           borderwidth=2,
                                           relief="sunken")
        self.image_label_frame.place(x=65, y=90, width=255, height=255)
        self.image_label = ttk.Label(self.image_label_frame)
        self.image_label.pack(fill="both", expand=True)

        self.heatmap_label_frame = ttk.Frame(self.root,
                                             borderwidth=2,
                                             relief="sunken")
        self.heatmap_label_frame.place(x=500, y=90, width=255, height=255)
        self.heatmap_label = ttk.Label(self.heatmap_label_frame)
        self.heatmap_label.pack(fill="both", expand=True)

        # Entradas de texto
        self.id_entry = ttk.Entry(self.root,
                                  textvariable=self.patient_id,
                                  width=20)
        self.result_text = tk.Text(self.root, width=15, height=1)
        self.proba_text = tk.Text(self.root, width=15, height=1)

        # Botones
        self.predict_button = ttk.Button(self.root, text="Predecir",
                                         state="disabled",
                                         command=self.run_model)
        self.load_button = ttk.Button(self.root, text="Cargar Imagen",
                                      command=self.load_image)
        self.clear_button = ttk.Button(self.root, text="Borrar",
                                       command=self.clear_ui)
        self.pdf_button = ttk.Button(self.root, text="Generar PDF",
                                     state="disabled",
                                     command=self.create_pdf)
        self.save_button = ttk.Button(self.root, text="Guardar en CSV",
                                      state="disabled",
                                      command=self.save_results_csv)

        # Posicionamiento de los widgets
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)

        self.load_button.place(x=70, y=460)
        self.predict_button.place(x=220, y=460)
        self.save_button.place(x=370, y=460)
        self.pdf_button.place(x=520, y=460)
        self.clear_button.place(x=670, y=460)

        self.id_entry.place(x=200, y=350)
        self.result_text.place(x=610, y=350, width=90, height=25)
        self.proba_text.place(x=610, y=400, width=90, height=25)

        self.id_entry.focus_set()

    def load_image(self):
        self.clear_ui(ask_confirm=False)
        self.filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("DICOM Files", "*.dcm"),
                       ("Image Files", "*.png;*.jpg;*.jpeg")]
        )

        if self.filepath:
            try:
                img_array, _ = read_dicom_file(self.filepath)
                self.original_image_pil = Image.fromarray(img_array).resize(
                    (250, 250), Image.LANCZOS)
                photo = ImageTk.PhotoImage(self.original_image_pil)

                self.image_label.config(image=photo)
                self.image_label.image = photo

                self.predict_button.config(state=tk.NORMAL)

            except Exception as e:
                self.clear_ui(ask_confirm=False)
                showinfo(title="Error", message=("No se pudo cargar la"
                                                 f"imagen.\nError: {e}"))

    def run_model(self):
        self.result_text.delete(1.0, tk.END)
        self.proba_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "Cargando...")
        self.root.update_idletasks()

        self.result_label, self.probability, heatmap_array = predict_neumonia(
            self.filepath
            )

        self.result_text.delete(1.0, tk.END)
        self.proba_text.delete(1.0, tk.END)

        if self.result_label:
            self.result_text.insert(tk.END, self.result_label)
            self.proba_text.insert(tk.END, f"{self.probability:.2f}%")

            self.heatmap_pil = Image.fromarray(heatmap_array).resize(
                (250, 250),
                Image.LANCZOS)
            photo_heatmap = ImageTk.PhotoImage(self.heatmap_pil)
            self.heatmap_label.config(image=photo_heatmap)
            self.heatmap_label.image = photo_heatmap

            self.save_button.config(state=tk.NORMAL)
            self.pdf_button.config(state=tk.NORMAL)
        else:
            showinfo(title="Error", message=("La predicción falló."
                                             " Revisa la consola."))

    def save_results_csv(self):
        if not self.result_label:
            showinfo(title="Guardar", message=("Primero debe realizar "
                                               "una predicción."))
            return

        # Creamos la carpeta 'historic' si no existe
        if not os.path.exists("reports/historic"):
            os.makedirs("reports/historic")

        # Generamos un nombre de archivo único
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostico_{self.patient_id.get()}_{timestamp}.csv"
        filepath = os.path.join("reports/historic", filename)

        with open(filepath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([self.patient_id.get(), self.result_label,
                             f"{self.probability:.2f}%",
                             os.path.basename(self.filepath)])

        showinfo(title="Guardar", message=("Los datos se guardaron "
                                           f"en:\n{filepath}"))

    def create_pdf(self):
        if not self.result_label:
            showinfo(title="PDF",
                     message="Primero debe realizar una predicción.")
            return

        try:
            # Creamos la carpeta 'reports' si no existe
            if not os.path.exists("reports"):
                os.makedirs("reports")

            # Generamos un nombre de archivo único para el PDF
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"reporte_{self.patient_id.get()}_{timestamp}.pdf"
            pdf_path = os.path.join("reports", pdf_filename)

            # La captura de pantalla es de la ventana completa
            import pyautogui
            screenshot = pyautogui.screenshot(region=(self.root.winfo_x(),
                                                      self.root.winfo_y(),
                                                      self.root.winfo_width(),
                                                      self.root.winfo_height())
                                              )

            # Guardamos la captura como un archivo temporal y
            # la convertimos a PDF
            screenshot_temp_path = "temp_screenshot.jpg"
            screenshot.save(screenshot_temp_path)

            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(screenshot_temp_path))

            os.remove(screenshot_temp_path)
            showinfo(title="PDF",
                     message=f"El PDF fue generado en:\n{pdf_path}")

        except Exception as e:
            showinfo(title="Error",
                     message=f"No se pudo generar el PDF.\nError: {e}")

    def clear_ui(self, ask_confirm=True):
        if ask_confirm:
            answer = askokcancel(title="Confirmación",
                                 message="Se borrarán todos los datos.",
                                 icon=WARNING)
        else:
            answer = True

        if answer:
            self.filepath = None
            self.patient_id.set("")
            self.result_text.delete(1.0, tk.END)
            self.proba_text.delete(1.0, tk.END)
            self.image_label.config(image='')
            self.heatmap_label.config(image='')

            self.predict_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.pdf_button.config(state=tk.DISABLED)

            if ask_confirm:
                showinfo(title="Borrar",
                         message="Los datos se borraron con éxito")


if __name__ == "__main__":
    app = App()
