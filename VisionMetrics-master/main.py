import customtkinter as ctk
from app import MetrologyApp

if __name__ == "__main__":
    root = ctk.CTk()
    app = MetrologyApp(root)
    root.mainloop()