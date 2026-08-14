from __future__ import annotations

import os
import sys
import tempfile
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, Menu
import customtkinter as ctk
from PIL import Image, ImageTk, ImageFont, ImageDraw

from measurements import (
    line_distance, angle_between, preview_angle,
    arc_canvas_points, arc_pil_params, format_distance,
)
from translations import TRANSLATIONS

def _resource_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_BTN_ACTIVE   = ("#3B8ED0", "#1F6AA5")
_BTN_INACTIVE = "transparent"
_BTN_BORDER   = ("gray50", "gray45")


class MetrologyApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Vision Metrics")
        self.root.minsize(1100, 650)

        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = 1700, 900
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        # ── Logo ───────────────────────────────────────────────────────────
        self._logo_pil = None
        try:
            self._logo_pil = Image.open(_resource_path("logo.png")).convert("RGBA")
            _ico_path = os.path.join(tempfile.gettempdir(), "visionmetrics.ico")
            self._logo_pil.save(_ico_path, format="ICO")
            self.root.iconbitmap(_ico_path)
        except Exception:
            pass

        # ── Language ───────────────────────────────────────────────────────
        self.lang = "en"
        self._tr_widgets: list[tuple] = []  # (widget, translation_key)

        # ── State ──────────────────────────────────────────────────────────
        self.line_color  = "#3B8ED0"
        self.text_color  = "#FFE000"
        self.point_color = "#E05C5C"

        self.zoom_level = 1.0
        self.offset_x   = 0
        self.offset_y   = 0
        self._pan_x     = None
        self._pan_y     = None

        self.image        = None
        self.image_tk     = None
        self.scale_factor = None

        self.calibration_points = []
        self.measurement_points = []
        self.lines        = []  # (p1, p2, pixel_dist)
        self.angles       = []  # (p1, p2, p3, angle_deg)
        self.arcs         = []  # list of arc-segment id lists
        self.texts        = []  # (x, y, text)
        self.action_stack = []

        self.current_text = ""
        self.adding_text  = False

        self.mode = tk.StringVar(value="line")
        self.unit = tk.StringVar(value="mm")

        self.show_lines_var  = tk.IntVar(value=1)
        self.show_points_var = tk.IntVar(value=1)
        self.show_angles_var = tk.IntVar(value=1)

        self._build_menu()
        self._build_layout()
        self._bind_keys()
        self._update_mode_buttons()

    # ══════════════════════════════════════════════════════════════════════
    # TRANSLATION
    # ══════════════════════════════════════════════════════════════════════

    def _t(self, key: str) -> str:
        return TRANSLATIONS[self.lang].get(key, TRANSLATIONS["en"][key])

    def _apply_language(self, lang: str):
        self.lang = lang
        self._build_menu()
        for widget, key in self._tr_widgets:
            try:
                widget.configure(text=self._t(key))
            except Exception:
                pass
        # Refresh dynamic status bar text
        mode = self.mode.get()
        self._st_mode.configure(
            text=f"{self._t('st_mode')}: {self._t(f'mode_{mode}')}")
        if self.scale_factor is None:
            self._st_calib.configure(text=self._t("st_not_calibrated"))

    def _on_language_change(self, choice: str):
        self._apply_language("cs" if choice == "Čeština" else "en")

    # ══════════════════════════════════════════════════════════════════════
    # MENU BAR
    # ══════════════════════════════════════════════════════════════════════

    def _build_menu(self):
        menubar = Menu(self.root)
        self.root.configure(menu=menubar)

        fm = Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self._t("menu_file"), menu=fm)
        fm.add_command(label=self._t("menu_open"), command=self.load_image)
        fm.add_command(label=self._t("menu_save"), command=self.save_image)
        fm.add_separator()
        fm.add_command(label=self._t("menu_exit"), command=self.root.quit)

        tm = Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self._t("menu_tools"), menu=tm)
        tm.add_command(label=self._t("menu_undo"),  command=self.undo_last_action)
        tm.add_command(label=self._t("menu_clear"), command=self.clear_measurements)
        tm.add_separator()
        for key, val in [("menu_line_mode", "line"), ("menu_angle_mode", "angle"),
                         ("menu_calibrate", "calibrate")]:
            tm.add_radiobutton(label=self._t(key), variable=self.mode, value=val,
                               command=lambda v=val: self._set_mode(v))

        vm = Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self._t("menu_view"), menu=vm)
        vm.add_command(label=self._t("menu_reset_view"),  command=self.reset_view)
        vm.add_command(label=self._t("menu_toggle_mode"), command=self.toggle_appearance)

        hm = Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self._t("menu_help"), menu=hm)
        hm.add_command(label=self._t("menu_about"), command=self._show_about)

    # ══════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ══════════════════════════════════════════════════════════════════════

    def _build_layout(self):
        self.sidebar = ctk.CTkScrollableFrame(self.root, width=260, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        right = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        right.pack(side="left", expand=True, fill="both")

        self.canvas = tk.Canvas(right, bg="#1c1c1e", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both", padx=8, pady=(8, 2))

        self._build_status_bar(right)
        self._build_sidebar()
        self._bind_canvas()

    def _build_status_bar(self, parent):
        bar = ctk.CTkFrame(parent, height=30, corner_radius=0)
        bar.pack(fill="x", padx=8, pady=(0, 6))
        bar.pack_propagate(False)

        def _sep():
            ctk.CTkLabel(bar, text=" │ ", text_color="gray40",
                         font=ctk.CTkFont(size=11)).pack(side="left")

        self._st_mode = ctk.CTkLabel(
            bar, text=f"{self._t('st_mode')}: {self._t('mode_line')}",
            width=110, anchor="w", font=ctk.CTkFont(size=11))
        self._st_mode.pack(side="left", padx=(8, 0))
        _sep()
        self._st_coords = ctk.CTkLabel(bar, text="X: —   Y: —", width=140, anchor="w",
                                       font=ctk.CTkFont(size=11))
        self._st_coords.pack(side="left")
        _sep()
        self._st_zoom = ctk.CTkLabel(bar, text="Zoom: 100%", width=90, anchor="w",
                                     font=ctk.CTkFont(size=11))
        self._st_zoom.pack(side="left")
        _sep()
        self._st_calib = ctk.CTkLabel(
            bar, text=self._t("st_not_calibrated"), width=200, anchor="w",
            text_color="#E05C5C", font=ctk.CTkFont(size=11))
        self._st_calib.pack(side="left")

    def _build_sidebar(self):
        self._sec_file()
        self._sec_mode()
        self._sec_text()
        self._sec_view()
        self._sec_colors()
        self._sec_units()
        self._sec_language()
        self._sec_history()

    # ── Sidebar helpers ────────────────────────────────────────────────────

    def _section(self, key: str) -> ctk.CTkFrame:
        wrap = ctk.CTkFrame(self.sidebar, corner_radius=10)
        wrap.pack(fill="x", padx=8, pady=(8, 0))
        lbl = ctk.CTkLabel(wrap, text=self._t(key),
                           font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
        lbl.pack(fill="x", padx=12, pady=(10, 4))
        self._tr_widgets.append((lbl, key))
        return wrap

    def _btn(self, parent, key: str, command, **kw) -> ctk.CTkButton:
        b = ctk.CTkButton(parent, text=self._t(key), command=command, height=30, **kw)
        b.pack(fill="x", padx=12, pady=2)
        self._tr_widgets.append((b, key))
        return b

    # ── Sidebar sections ───────────────────────────────────────────────────

    def _sec_file(self):
        f = self._section("sec_file")
        self._btn(f, "btn_load", self.load_image)
        self._btn(f, "btn_save", self.save_image,
                  fg_color=_BTN_INACTIVE, border_width=1, border_color=_BTN_BORDER)
        ctk.CTkFrame(f, height=6, fg_color="transparent").pack()

    def _sec_mode(self):
        f = self._section("sec_mode")
        self._mode_btns: dict[str, ctk.CTkButton] = {}
        for key, val in [("btn_line", "line"), ("btn_angle", "angle"),
                         ("btn_calibrate", "calibrate")]:
            b = ctk.CTkButton(f, text=self._t(key), height=30,
                              command=lambda v=val: self._set_mode(v),
                              fg_color=_BTN_INACTIVE, border_width=1,
                              border_color=_BTN_BORDER)
            b.pack(fill="x", padx=12, pady=2)
            self._mode_btns[val] = b
            self._tr_widgets.append((b, key))

        ctk.CTkFrame(f, height=4, fg_color="transparent").pack()
        ctk.CTkFrame(f, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=4)

        self._btn(f, "btn_undo", self.undo_last_action,
                  fg_color=_BTN_INACTIVE, border_width=1, border_color=_BTN_BORDER)
        self._btn(f, "btn_clear", self.clear_measurements,
                  fg_color=("#8B2222", "#7A1E1E"), hover_color=("#A03030", "#8B2525"))
        ctk.CTkFrame(f, height=6, fg_color="transparent").pack()

    def _sec_text(self):
        f = self._section("sec_annotations")
        self._btn(f, "btn_add_text", self._start_text_flow,
                  fg_color=_BTN_INACTIVE, border_width=1, border_color=_BTN_BORDER)
        ctk.CTkFrame(f, height=6, fg_color="transparent").pack()

    def _sec_view(self):
        f = self._section("sec_view")
        for key, var in [("chk_lines",  self.show_lines_var),
                         ("chk_points", self.show_points_var),
                         ("chk_angles", self.show_angles_var)]:
            cb = ctk.CTkCheckBox(f, text=self._t(key), variable=var,
                                 command=self.redraw_measurements)
            cb.pack(anchor="w", padx=14, pady=2)
            self._tr_widgets.append((cb, key))
        self._btn(f, "btn_reset_view", self.reset_view,
                  fg_color=_BTN_INACTIVE, border_width=1, border_color=_BTN_BORDER)
        ctk.CTkFrame(f, height=6, fg_color="transparent").pack()

    def _sec_colors(self):
        f = self._section("sec_colors")
        self._btn(f, "btn_line_color",  self.change_line_color,
                  fg_color=_BTN_INACTIVE, border_width=1, border_color=_BTN_BORDER)
        self._btn(f, "btn_text_color",  self.change_text_color,
                  fg_color=_BTN_INACTIVE, border_width=1, border_color=_BTN_BORDER)
        self._btn(f, "btn_point_color", self.change_point_color,
                  fg_color=_BTN_INACTIVE, border_width=1, border_color=_BTN_BORDER)
        ctk.CTkFrame(f, height=6, fg_color="transparent").pack()

    def _sec_units(self):
        f = self._section("sec_units")
        ctk.CTkOptionMenu(f, variable=self.unit,
                          values=["mm", "cm", "in", "px"],
                          command=lambda _: self.redraw_measurements()
                          ).pack(fill="x", padx=12, pady=(2, 10))

    def _sec_language(self):
        f = self._section("sec_language")
        ctk.CTkOptionMenu(
            f, values=["English", "Čeština"],
            command=self._on_language_change
        ).pack(fill="x", padx=12, pady=(2, 10))

    def _sec_history(self):
        f = self._section("sec_history")
        self.history_list = tk.Listbox(
            f, height=10, relief="flat", bd=0, highlightthickness=0,
            bg="#2b2b2b", fg="white", selectbackground="#1F6AA5",
            selectforeground="white", font=("Consolas", 10)
        )
        self.history_list.pack(fill="x", padx=12, pady=(2, 10))

    # ══════════════════════════════════════════════════════════════════════
    # BINDINGS
    # ══════════════════════════════════════════════════════════════════════

    def _bind_canvas(self):
        self.canvas.bind("<Button-1>",        self.on_click)
        self.canvas.bind("<Motion>",          self.on_motion)
        self.canvas.bind("<MouseWheel>",      self.on_zoom)
        self.canvas.bind("<ButtonPress-2>",   self._pan_start)
        self.canvas.bind("<B2-Motion>",       self._pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._pan_stop)

    def _bind_keys(self):
        r = self.root
        r.bind("<Control-o>", lambda _: self.load_image())
        r.bind("<Control-s>", lambda _: self.save_image())
        r.bind("<Control-z>", lambda _: self.undo_last_action())
        r.bind("l", lambda _: self._set_mode("line"))
        r.bind("a", lambda _: self._set_mode("angle"))
        r.bind("c", lambda _: self._set_mode("calibrate"))
        r.bind("r", lambda _: self.reset_view())
        r.bind("<Escape>", lambda _: self._cancel_input())

    # ══════════════════════════════════════════════════════════════════════
    # MODE MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def _set_mode(self, mode: str):
        self._cancel_input()
        self.mode.set(mode)
        self._update_mode_buttons()
        self._st_mode.configure(
            text=f"{self._t('st_mode')}: {self._t(f'mode_{mode}')}")

    def _update_mode_buttons(self):
        active = self.mode.get()
        for key, btn in self._mode_btns.items():
            if key == active:
                btn.configure(fg_color=_BTN_ACTIVE, border_width=0)
            else:
                btn.configure(fg_color=_BTN_INACTIVE, border_width=1,
                              border_color=_BTN_BORDER)

    def _cancel_input(self):
        self.measurement_points.clear()
        self.calibration_points.clear()
        self.adding_text = False
        self.redraw_measurements()

    # ══════════════════════════════════════════════════════════════════════
    # IMAGE LOAD / DISPLAY / SAVE
    # ══════════════════════════════════════════════════════════════════════

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        self.image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            messagebox.showerror(self._t("title_error"), self._t("err_load_fail"))
            return
        self.zoom_level = 1.0
        self.offset_x = self.offset_y = 0
        self.scale_factor = None
        self._st_calib.configure(text=self._t("st_not_calibrated"), text_color="#E05C5C")
        self.display_image()

    def display_image(self):
        if self.image is None:
            return
        h, w = self.image.shape[:2]

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1:
            cw = self.root.winfo_width() - 280
            ch = self.root.winfo_height() - 60

        img_x0 = max(0.0, -self.offset_x / self.zoom_level)
        img_y0 = max(0.0, -self.offset_y / self.zoom_level)
        img_x1 = min(float(w), (cw - self.offset_x) / self.zoom_level)
        img_y1 = min(float(h), (ch - self.offset_y) / self.zoom_level)

        if img_x1 <= img_x0 or img_y1 <= img_y0:
            self.canvas.delete("all")
            self.redraw_measurements()
            return

        cx0, cy0 = int(img_x0), int(img_y0)
        cx1 = min(w, int(img_x1) + 1)
        cy1 = min(h, int(img_y1) + 1)
        crop = self.image[cy0:cy1, cx0:cx1]

        out_w = max(1, int((cx1 - cx0) * self.zoom_level))
        out_h = max(1, int((cy1 - cy0) * self.zoom_level))
        interp = cv2.INTER_NEAREST if self.zoom_level >= 1.0 else cv2.INTER_AREA
        resized = cv2.resize(crop, (out_w, out_h), interpolation=interp)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.image_tk = ImageTk.PhotoImage(Image.fromarray(rgb))

        draw_x = int(cx0 * self.zoom_level + self.offset_x)
        draw_y = int(cy0 * self.zoom_level + self.offset_y)

        self.canvas.delete("all")
        self.canvas.create_image(draw_x, draw_y, anchor="nw", image=self.image_tk)
        self.redraw_measurements()

    def save_image(self):
        if self.image is None:
            messagebox.showwarning(self._t("title_warning"), self._t("err_no_image"))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return

        pil_img = Image.fromarray(cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            font = ImageFont.load_default()

        for start, end, px_dist in self.lines:
            s = (int(start[0]), int(start[1]))
            e = (int(end[0]),   int(end[1]))
            draw.line([s, e], fill=self.line_color, width=2)
            mid = ((s[0]+e[0])//2, (s[1]+e[1])//2)
            draw.text(mid, self._fmt(px_dist), fill=self.text_color, font=font)

        for p1, p2, p3, angle in self.angles:
            p1p = (int(p1[0]), int(p1[1]))
            p2p = (int(p2[0]), int(p2[1]))
            p3p = (int(p3[0]), int(p3[1]))
            draw.line([p2p, p1p], fill=self.line_color, width=2)
            draw.line([p2p, p3p], fill=self.line_color, width=2)
            sa, ea, r = arc_pil_params(p2p, p1p, p3p)
            bbox = [(p2p[0]-r, p2p[1]-r), (p2p[0]+r, p2p[1]+r)]
            draw.arc(bbox, start=sa, end=ea, fill=self.line_color, width=2)
            draw.text((p2p[0]+20, p2p[1]-20), f"{angle:.2f}°",
                      fill=self.text_color, font=font)

        for x, y, text in self.texts:
            draw.text((x, y), text, fill=self.text_color, font=font)

        out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, out)

    # ══════════════════════════════════════════════════════════════════════
    # CLICK HANDLING
    # ══════════════════════════════════════════════════════════════════════

    def on_click(self, event):
        pt = [(event.x - self.offset_x) / self.zoom_level,
              (event.y - self.offset_y) / self.zoom_level]

        # Snap second line point to 45° increments when Shift is held
        if (event.state & 0x1) and self.mode.get() == "line" and len(self.measurement_points) == 1:
            p1_canvas = self._sop(self.measurement_points[0])
            sx, sy = self._snap_to_axis(p1_canvas, event.x, event.y)
            pt = [(sx - self.offset_x) / self.zoom_level,
                  (sy - self.offset_y) / self.zoom_level]

        if self.adding_text:
            if self.image is not None and self.current_text:
                x, y = int(pt[0]), int(pt[1])
                self.texts.append((x, y, self.current_text))
                self.action_stack.append({'type': 'text', 'data': (x, y, self.current_text)})
                self._hist("text", text=self.current_text)
                self.adding_text = False
                self.redraw_measurements()
            return

        mode = self.mode.get()

        if mode == "calibrate":
            self.calibration_points.append(pt)
            self.redraw_measurements()
            if len(self.calibration_points) == 2:
                self._do_calibrate()

        elif mode == "line" and len(self.measurement_points) < 2:
            self.measurement_points.append(pt)
            self.action_stack.append({'type': 'point', 'data': pt})
            self._hist("point", x=pt[0], y=pt[1])
            if len(self.measurement_points) == 2:
                self._finish_line()
            else:
                self.redraw_measurements()

        elif mode == "angle" and len(self.measurement_points) < 3:
            self.measurement_points.append(pt)
            self.action_stack.append({'type': 'point', 'data': pt})
            if len(self.measurement_points) == 3:
                self._finish_angle()
            else:
                self.redraw_measurements()

    def _finish_line(self):
        p1, p2 = self.measurement_points[:2]
        px = line_distance(p1, p2)
        if px == 0:
            messagebox.showerror(self._t("title_error"), self._t("err_identical"))
            self.measurement_points.clear()
            return
        line = (p1, p2, px)
        self.lines.append(line)
        self.action_stack.append({'type': 'line', 'line': line})
        self._hist("line", px=px)
        self.measurement_points.clear()
        self.redraw_measurements()

    def _finish_angle(self):
        p1, p2, p3 = self.measurement_points[:3]
        deg = angle_between(p1, p2, p3)
        angle = (p1, p2, p3, deg)
        self.angles.append(angle)
        arc = self._draw_arc(self._sop(p2), self._sop(p1), self._sop(p3), record=True)
        self.action_stack.append({'type': 'angle', 'angle': angle, 'arc': arc})
        self._hist("angle", deg=deg)
        self.measurement_points.clear()
        self.redraw_measurements()

    # ══════════════════════════════════════════════════════════════════════
    # CALIBRATION
    # ══════════════════════════════════════════════════════════════════════

    def _do_calibrate(self):
        p1, p2 = self.calibration_points
        px = line_distance(p1, p2)
        if px == 0:
            messagebox.showerror(self._t("title_error"), self._t("err_overlap"))
            self.calibration_points.clear()
            return

        dlg = ctk.CTkToplevel(self.root)
        dlg.title(self._t("dlg_calib_title"))
        dlg.geometry("340x180")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.lift()

        ctk.CTkLabel(dlg, text=self._t("dlg_calib_header"),
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(18, 4))
        ctk.CTkLabel(dlg, text=f"{self._t('dlg_calib_px')}: {px:.1f} px",
                     font=ctk.CTkFont(size=12), text_color="gray").pack()

        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(pady=8)
        ctk.CTkLabel(row, text=self._t("dlg_calib_known")).pack(side="left", padx=(0, 6))
        entry = ctk.CTkEntry(row, width=110, placeholder_text="e.g. 25.4")
        entry.pack(side="left", padx=2)
        ctk.CTkLabel(row, text=self.unit.get()).pack(side="left", padx=(4, 0))

        def confirm():
            try:
                known = float(entry.get())
                prev_pts   = self.calibration_points[:]
                prev_scale = self.scale_factor
                self.scale_factor = known / px
                self.calibration_points.clear()
                self.action_stack.append({
                    'type': 'calibration',
                    'previous_points': prev_pts,
                    'previous_scale':  prev_scale,
                })
                self._st_calib.configure(
                    text=f"● {self.scale_factor:.4f} {self.unit.get()}/px",
                    text_color="#4CAF50")
                dlg.destroy()
                self.redraw_measurements()
            except ValueError:
                messagebox.showerror(self._t("title_error"), self._t("err_numeric"))

        entry.bind("<Return>", lambda _: confirm())
        ctk.CTkButton(dlg, text=self._t("btn_set_scale"), command=confirm).pack(pady=4)

    # ══════════════════════════════════════════════════════════════════════
    # TEXT ANNOTATION
    # ══════════════════════════════════════════════════════════════════════

    def _start_text_flow(self):
        if self.image is None:
            messagebox.showwarning(self._t("title_warning"), self._t("err_load_first"))
            return
        dlg = ctk.CTkToplevel(self.root)
        dlg.title(self._t("dlg_text_title"))
        dlg.geometry("320x150")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.lift()

        ctk.CTkLabel(dlg, text=self._t("dlg_text_label"),
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(16, 4))
        entry = ctk.CTkEntry(dlg, width=260,
                             placeholder_text=self._t("dlg_text_ph"))
        entry.pack(pady=4)
        entry.focus()

        def confirm():
            t = entry.get().strip()
            if t:
                self.current_text = t
                self.adding_text  = True
                dlg.destroy()
                self._st_mode.configure(text=self._t("st_click_place"))

        entry.bind("<Return>", lambda _: confirm())
        ctk.CTkButton(dlg, text=self._t("btn_place_text"), command=confirm).pack(pady=8)

    # ══════════════════════════════════════════════════════════════════════
    # UNDO / CLEAR
    # ══════════════════════════════════════════════════════════════════════

    def undo_last_action(self):
        if not self.action_stack:
            return
        last = self.action_stack.pop()
        t = last['type']

        def _pop_points(n):
            for _ in range(n):
                if self.action_stack and self.action_stack[-1]['type'] == 'point':
                    self.action_stack.pop()

        if t == 'line' and self.lines:
            self.lines.pop()
            _pop_points(2)
            self._hist_pop()
        elif t == 'angle' and self.angles:
            self.angles.pop()
            if self.arcs:
                for seg in self.arcs.pop():
                    self.canvas.delete(seg)
            _pop_points(3)
            self._hist_pop()
        elif t == 'point' and self.measurement_points:
            self.measurement_points.pop()
        elif t == 'calibration':
            self.calibration_points = last.get('previous_points', [])
            self.scale_factor       = last.get('previous_scale', None)
            if self.scale_factor is None:
                self._st_calib.configure(text=self._t("st_not_calibrated"),
                                         text_color="#E05C5C")
            else:
                self._st_calib.configure(
                    text=f"● {self.scale_factor:.4f} {self.unit.get()}/px",
                    text_color="#4CAF50")
        elif t == 'text' and self.texts:
            self.texts.pop()
            self._hist_pop()

        self.redraw_measurements()

    def clear_measurements(self):
        self.calibration_points.clear()
        self.measurement_points.clear()
        self.lines.clear()
        self.angles.clear()
        self.texts.clear()
        self.arcs.clear()
        self.action_stack.clear()
        self.history_list.delete(0, "end")
        self.display_image()

    # ══════════════════════════════════════════════════════════════════════
    # REDRAW
    # ══════════════════════════════════════════════════════════════════════

    def redraw_measurements(self):
        self.canvas.delete("measurement")
        self.canvas.delete("preview")

        for x, y, text in self.texts:
            sx, sy = self._sop([x, y])
            self.canvas.create_text(sx, sy, text=text, fill=self.text_color,
                                    font=("Arial", 12), tags="measurement")

        if self.show_points_var.get():
            for pt in self.calibration_points + self.measurement_points:
                sx, sy = self._sop(pt)
                self.canvas.create_oval(sx-5, sy-5, sx+5, sy+5,
                                        fill=self.point_color, outline="white",
                                        width=1, tags="measurement")

        if self.show_lines_var.get():
            for i, (start, end, px) in enumerate(self.lines):
                ss, se = self._sop(start), self._sop(end)
                tag = f"ln{i}"
                self.canvas.create_line(*ss, *se, fill=self.line_color,
                                        width=2, tags=(tag, "measurement"))
                mx, my = (ss[0]+se[0])//2, (ss[1]+se[1])//2
                self._lbl(mx, my - 12, self._fmt(px))
                self.canvas.tag_bind(tag, "<Enter>",
                    lambda e, l=self._fmt(px): self._tooltip(e.x, e.y, l))

        if self.show_angles_var.get():
            for i, (p1, p2, p3, deg) in enumerate(self.angles):
                sp1, sp2, sp3 = self._sop(p1), self._sop(p2), self._sop(p3)
                tag = f"ag{i}"
                self.canvas.create_line(*sp2, *sp1, fill=self.line_color,
                                        width=2, tags=(tag, "measurement"))
                self.canvas.create_line(*sp2, *sp3, fill=self.line_color,
                                        width=2, tags=(tag, "measurement"))
                self._draw_arc(sp2, sp1, sp3, record=False)
                self._lbl(sp2[0], sp2[1] - 22, f"{deg:.2f}°")
                self.canvas.tag_bind(tag, "<Enter>",
                    lambda e, d=deg: self._tooltip(e.x, e.y, f"{d:.2f}°"))

    # ══════════════════════════════════════════════════════════════════════
    # LIVE PREVIEW
    # ══════════════════════════════════════════════════════════════════════

    def _snap_to_axis(self, p1: tuple, ex: int, ey: int) -> tuple[int, int]:
        from math import atan2, pi, sqrt, cos, sin
        dx, dy = ex - p1[0], ey - p1[1]
        angle = atan2(dy, dx)
        snapped = round(angle / (pi / 4)) * (pi / 4)
        dist = sqrt(dx * dx + dy * dy)
        return (int(p1[0] + dist * cos(snapped)), int(p1[1] + dist * sin(snapped)))

    def on_motion(self, event):
        mode = self.mode.get()
        cursors = {"calibrate": "crosshair", "line": "crosshair",
                   "angle": "crosshair", "text": "xterm"}
        self.canvas.config(cursor=cursors.get(mode, "arrow"))
        self.canvas.delete("preview")

        if self.image is not None:
            ix = (event.x - self.offset_x) / self.zoom_level
            iy = (event.y - self.offset_y) / self.zoom_level
            self._st_coords.configure(text=f"X: {ix:.1f}   Y: {iy:.1f}")

        if mode == "line" and len(self.measurement_points) == 1:
            p1 = self._sop(self.measurement_points[0])
            tx, ty = event.x, event.y
            if event.state & 0x1:  # Shift held — snap to 45° increments
                tx, ty = self._snap_to_axis(p1, event.x, event.y)
            self.canvas.create_line(*p1, tx, ty,
                                    fill=self.line_color, width=2, dash=(6, 4), tags="preview")
            from math import sqrt
            px = sqrt((tx - p1[0])**2 + (ty - p1[1])**2) / self.zoom_level
            self._plbl((p1[0]+tx)//2, (p1[1]+ty)//2 - 14, self._fmt(px))

        elif mode == "angle":
            if len(self.measurement_points) == 1:
                p1 = self._sop(self.measurement_points[0])
                self.canvas.create_line(*p1, event.x, event.y,
                                        fill=self.line_color, width=2, dash=(6, 4), tags="preview")
            elif len(self.measurement_points) == 2:
                p1 = self._sop(self.measurement_points[0])
                p2 = self._sop(self.measurement_points[1])
                self.canvas.create_line(*p2, *p1, fill=self.line_color, width=2, tags="preview")
                self.canvas.create_line(*p2, event.x, event.y,
                                        fill=self.line_color, width=2, dash=(6, 4), tags="preview")
                ang = preview_angle(p1, p2, event.x, event.y)
                if ang is not None:
                    self._plbl(p2[0], p2[1] - 22, f"{ang:.1f}°")

        elif mode == "calibrate" and len(self.calibration_points) == 1:
            p1 = self._sop(self.calibration_points[0])
            self.canvas.create_line(*p1, event.x, event.y,
                                    fill="#4CAF50", width=2, dash=(6, 4), tags="preview")
            from math import sqrt
            px = sqrt((event.x-p1[0])**2 + (event.y-p1[1])**2) / self.zoom_level
            self._plbl((p1[0]+event.x)//2, (p1[1]+event.y)//2 - 14, f"{px:.1f} px")

    # ══════════════════════════════════════════════════════════════════════
    # PAN / ZOOM
    # ══════════════════════════════════════════════════════════════════════

    def _pan_start(self, event):
        self._pan_x, self._pan_y = event.x, event.y
        self.canvas.config(cursor="fleur")

    def _pan_move(self, event):
        if self._pan_x is not None:
            self.offset_x += event.x - self._pan_x
            self.offset_y += event.y - self._pan_y
            self._pan_x, self._pan_y = event.x, event.y
            self.display_image()

    def _pan_stop(self, _event):
        self._pan_x = self._pan_y = None
        self.canvas.config(cursor="arrow")

    def on_zoom(self, event):
        old_zoom = self.zoom_level
        self.zoom_level = max(0.05, min(old_zoom * (1.1 if event.delta > 0 else 0.9), 20))
        self.offset_x = event.x - (event.x - self.offset_x) * (self.zoom_level / old_zoom)
        self.offset_y = event.y - (event.y - self.offset_y) * (self.zoom_level / old_zoom)
        self._st_zoom.configure(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.display_image()

    def reset_view(self):
        self.zoom_level = 1.0
        self.offset_x = self.offset_y = 0
        self._st_zoom.configure(text="Zoom: 100%")
        self.display_image()

    # ══════════════════════════════════════════════════════════════════════
    # COLOR PICKERS
    # ══════════════════════════════════════════════════════════════════════

    def change_line_color(self):
        c = colorchooser.askcolor(title="Line Color", color=self.line_color)[1]
        if c:
            self.line_color = c
            self.redraw_measurements()

    def change_text_color(self):
        c = colorchooser.askcolor(title="Text Color", color=self.text_color)[1]
        if c:
            self.text_color = c
            self.redraw_measurements()

    def change_point_color(self):
        c = colorchooser.askcolor(title="Point Color", color=self.point_color)[1]
        if c:
            self.point_color = c
            self.redraw_measurements()

    # ══════════════════════════════════════════════════════════════════════
    # APPEARANCE
    # ══════════════════════════════════════════════════════════════════════

    def toggle_appearance(self):
        new = "light" if ctk.get_appearance_mode() == "Dark" else "dark"
        ctk.set_appearance_mode(new)
        self.canvas.configure(bg="#1c1c1e" if new == "dark" else "#d0d0d0")
        bg = "#2b2b2b" if new == "dark" else "#f0f0f0"
        fg = "white"   if new == "dark" else "black"
        self.history_list.configure(bg=bg, fg=fg)

    # ══════════════════════════════════════════════════════════════════════
    # ARC DRAWING
    # ══════════════════════════════════════════════════════════════════════

    def _draw_arc(self, center, start, end, radius=50, record=True):
        pts = arc_canvas_points(center, start, end, radius=radius)
        segs = [
            self.canvas.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                                    fill=self.line_color, width=2, tags="measurement")
            for i in range(len(pts) - 1)
        ]
        if record:
            self.arcs.append(segs)
        return segs

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _sop(self, point) -> tuple[int, int]:
        return (int(point[0] * self.zoom_level + self.offset_x),
                int(point[1] * self.zoom_level + self.offset_y))

    def _fmt(self, px_dist: float) -> str:
        return format_distance(px_dist, self.scale_factor, self.unit.get())

    def _lbl(self, x, y, text):
        self.canvas.create_text(x, y, text=text, fill=self.text_color,
                                font=("Arial", 10, "bold"), tags="measurement")

    def _plbl(self, x, y, text):
        self.canvas.create_text(x, y, text=text, fill=self.text_color,
                                font=("Arial", 10, "bold"), tags="preview")

    def _tooltip(self, x, y, text):
        self.canvas.create_text(x+12, y+12, text=text, fill="white",
                                font=("Arial", 10), tags="tooltip", anchor="nw")
        self.canvas.after(1500, lambda: self.canvas.delete("tooltip"))

    def _hist(self, kind, **kw):
        if kind == "point":
            self.history_list.insert("end", f"  ● ({kw['x']:.1f}, {kw['y']:.1f})")
        elif kind == "line":
            self.history_list.insert("end", f"  ─ {self._fmt(kw['px'])}")
        elif kind == "angle":
            self.history_list.insert("end", f"  ∠ {kw['deg']:.2f}°")
        elif kind == "text":
            self.history_list.insert("end", f"  T \"{kw['text']}\"")
        self.history_list.yview("end")

    def _hist_pop(self):
        if self.history_list.size():
            self.history_list.delete("end")

    def _show_about(self):
        dlg = ctk.CTkToplevel(self.root)
        dlg.title(self._t("dlg_about_title"))
        dlg.geometry("360x220")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.lift()
        ctk.CTkLabel(dlg, text="Vision Metrics",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(28, 6))
        ctk.CTkLabel(dlg, text=self._t("dlg_about_sub"),
                     font=ctk.CTkFont(size=12)).pack()
        ctk.CTkLabel(dlg, text=self._t("dlg_about_feat"),
                     font=ctk.CTkFont(size=11), text_color="gray").pack(pady=4)
        ctk.CTkLabel(dlg, text=self._t("dlg_about_keys"),
                     font=ctk.CTkFont(size=10), text_color="gray").pack(pady=2)
        ctk.CTkButton(dlg, text=self._t("btn_close"),
                      command=dlg.destroy, width=100).pack(pady=16)
