import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw
import os

# ─────────────────────────────────────────────────────────────────
#  LUT PRESETS
# ─────────────────────────────────────────────────────────────────

def make_lut(stops):
    lut = np.zeros((256, 3), dtype=np.uint8)
    stops = sorted(stops, key=lambda s: s[0])
    for i in range(len(stops) - 1):
        g0, b0, gr0, r0 = stops[i]
        g1, b1, gr1, r1 = stops[i + 1]
        for g in range(g0, g1 + 1):
            t = (g - g0) / max(g1 - g0, 1)
            lut[g] = [int(b0+(b1-b0)*t), int(gr0+(gr1-gr0)*t), int(r0+(r1-r0)*t)]
    return lut

PRESETS = {
    "Vintage Warm":  make_lut([(0,20,15,8),(60,40,55,90),(120,60,110,160),(180,110,165,205),(220,170,205,230),(255,240,238,248)]),
    "Sepia Classic": make_lut([(0,14,12,10),(60,35,30,20),(120,90,75,55),(180,148,128,100),(220,195,172,138),(255,232,214,186)]),
    "Nature Green":  make_lut([(0,8,18,8),(60,25,65,28),(120,45,115,55),(180,85,162,82),(220,130,200,105),(255,195,232,178)]),
    "Ocean Blue":    make_lut([(0,20,8,5),(60,95,45,18),(120,155,95,38),(180,198,148,75),(220,220,190,130),(255,242,232,200)]),
    "Golden Hour":   make_lut([(0,10,5,18),(60,20,30,110),(120,30,100,210),(180,50,170,245),(220,110,210,255),(255,200,238,255)]),
    "Sunset Dusk":   make_lut([(0,25,5,35),(60,70,18,120),(120,40,60,210),(160,30,120,235),(200,70,170,245),(255,200,225,255)]),
    "Noir Film":     make_lut([(0,10,10,10),(80,55,52,50),(160,120,115,110),(220,190,185,180),(255,240,238,235)]),
    "Cyberpunk":     make_lut([(0,30,0,18),(60,175,0,75),(120,250,0,145),(170,195,45,248),(210,100,145,255),(255,205,232,255)]),
    "Autumn Leaves": make_lut([(0,10,8,12),(60,20,45,95),(120,30,90,185),(170,40,140,225),(210,70,185,240),(255,180,225,255)]),
    "Arctic Ice":    make_lut([(0,18,12,10),(80,100,80,55),(150,185,165,120),(210,230,218,185),(255,250,245,235)]),
}

PRESET_NAMES = list(PRESETS.keys())
DOT_COLORS = {
    "Vintage Warm":"#f59e0b","Sepia Classic":"#a78b55","Nature Green":"#22c55e",
    "Ocean Blue":"#3b82f6","Golden Hour":"#facc15","Sunset Dusk":"#f43f5e",
    "Noir Film":"#94a3b8","Cyberpunk":"#c026d3","Autumn Leaves":"#ea580c","Arctic Ice":"#67e8f9",
}

BG=  "#0a0c14"; PANEL="#11141f"; PANEL2="#181c2a"
ACCENT="#f97316"; ACCENT2="#818cf8"; FG="#e2e8f0"; MUTED="#475569"
SUCCESS="#10b981"; PURPLE="#7c3aed"

# ─────────────────────────────────────────────────────────────────
#  CORE PROCESSING
# ─────────────────────────────────────────────────────────────────

def apply_lut(gray, lut):
    result = np.zeros((*gray.shape, 3), dtype=np.uint8)
    result[:,:,0] = lut[gray,0]; result[:,:,1] = lut[gray,1]; result[:,:,2] = lut[gray,2]
    return result

def colorize(img_bgr, preset, intensity, brightness, contrast, saturation, sharpness, denoise, auto_enhance):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if auto_enhance: gray = cv2.equalizeHist(gray)
    if denoise:      gray = cv2.fastNlMeansDenoising(gray, h=10)
    colored  = apply_lut(gray, PRESETS[preset])
    alpha    = intensity / 100.0
    gray3    = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended  = cv2.addWeighted(colored, alpha, gray3, 1-alpha, 0)
    pil = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Brightness(pil).enhance(brightness/100)
    pil = ImageEnhance.Contrast(pil).enhance(contrast/100)
    pil = ImageEnhance.Color(pil).enhance(saturation/100)
    if sharpness > 100:
        factor = (sharpness-100)/100
        pil = pil.filter(ImageFilter.UnsharpMask(radius=2,percent=int(factor*150),threshold=3))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def fit(img, max_w, max_h):
    h,w = img.shape[:2]
    scale = min(max_w/w, max_h/h)
    if scale >= 1.0: return img
    return cv2.resize(img,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)

def make_histogram(gray, width=206, height=75):
    hist = cv2.calcHist([gray],[0],None,[64],[0,256]).flatten()
    hist = hist/(hist.max()+1e-6)
    img  = np.full((height,width,3),25,dtype=np.uint8)
    bw   = width//64
    for i,v in enumerate(hist):
        bh = int(v*(height-4))
        shade = int(50+i*3)
        cv2.rectangle(img,(i*bw,height-bh),(i*bw+bw-1,height),(shade,shade+20,shade+40),-1)
    return img

# ─────────────────────────────────────────────────────────────────
#  COMPARE RENDERER  — single canvas, drag divider
# ─────────────────────────────────────────────────────────────────

def render_compare(orig_bgr, proc_bgr, cw, ch, divider_x):
    """Returns a (ch x cw x 3) uint8 RGB array with left=B&W, right=color."""
    # Fit both to full canvas size
    def to_rgb_pil(img):
        r = fit(img, cw, ch)
        full = Image.new("RGB",(cw,ch),(10,12,20))
        rh,rw = r.shape[:2]
        ox=(cw-rw)//2; oy=(ch-rh)//2
        full.paste(Image.fromarray(cv2.cvtColor(r,cv2.COLOR_BGR2RGB)),(ox,oy))
        return np.array(full)

    orig_arr = to_rgb_pil(orig_bgr)
    proc_arr = to_rgb_pil(proc_bgr)

    divider_x = max(2, min(cw-2, divider_x))
    combined  = orig_arr.copy()
    combined[:, divider_x:] = proc_arr[:, divider_x:]

    # Orange divider line
    combined[:, divider_x-1:divider_x+2] = [249, 115, 22]
    # Arrow handles on divider
    mid = ch // 2
    for dy in range(-12, 13):
        if 0 <= mid+dy < ch:
            combined[mid+dy, divider_x-1:divider_x+2] = [255, 255, 255]

    # Labels
    pil = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil)
    lx = divider_x // 2
    rx = divider_x + (cw - divider_x) // 2
    draw.text((lx-1,15), "◀ ORIGINAL",  fill="#000000")
    draw.text((lx,  14), "◀ ORIGINAL",  fill="#f97316")
    draw.text((rx-1,15), "COLORIZED ▶", fill="#000000")
    draw.text((rx,  14), "COLORIZED ▶", fill="#818cf8")
    return np.array(pil)

# ─────────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("B&W Image Colorization — Advanced Dashboard")
        self.root.geometry("1400x820")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.orig_bgr     = None   # grayscale stored as BGR
        self.proc_bgr     = None   # colorized
        self.undo_stack   = []
        self.compare_on   = False
        self.div_x        = None   # divider position (pixels)
        self.dragging     = False

        self.preset_var   = tk.StringVar(value=PRESET_NAMES[0])
        self.intensity    = tk.IntVar(value=85)
        self.brightness   = tk.IntVar(value=100)
        self.contrast     = tk.IntVar(value=105)
        self.saturation   = tk.IntVar(value=120)
        self.sharpness    = tk.IntVar(value=110)
        self.denoise      = tk.BooleanVar(value=False)
        self.auto_enhance = tk.BooleanVar(value=False)

        self._build()

    # ── Layout ────────────────────────────────────────────────────
    def _build(self):
        top = tk.Frame(self.root, bg=PANEL, height=52)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="B&W Colorization  ·  Advanced Dashboard",
                 font=("Consolas",13,"bold"), fg=ACCENT, bg=PANEL
                 ).pack(side="left", padx=20, pady=14)
        tk.Label(top, text="OpenCV + PIL  |  Mini Project",
                 font=("Consolas",9), fg=MUTED, bg=PANEL
                 ).pack(side="right", padx=20)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        sidebar = tk.Frame(body, bg=PANEL, width=250)
        sidebar.pack(side="left", fill="y", padx=(0,8))
        sidebar.pack_propagate(False)
        self._sidebar(sidebar)

        center = tk.Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True)
        self._center(center)

        right = tk.Frame(body, bg=PANEL, width=230)
        right.pack(side="right", fill="y", padx=(8,0))
        right.pack_propagate(False)
        self._right(right)

        self.status = tk.StringVar(value="Upload a B&W photo to begin.")
        tk.Label(self.root, textvariable=self.status,
                 font=("Consolas",9), fg=MUTED, bg=BG, anchor="w"
                 ).pack(fill="x", padx=14, pady=(0,4))

    def _sec(self, parent, t):
        tk.Frame(parent, bg=PANEL2, height=1).pack(fill="x", padx=10, pady=(14,0))
        tk.Label(parent, text=t, font=("Consolas",8,"bold"),
                 fg=ACCENT, bg=PANEL, anchor="w"
                 ).pack(fill="x", padx=14, pady=(6,4))

    def _btn(self, parent, text, cmd, bg, fg):
        tk.Button(parent, text=text, command=cmd,
                  font=("Consolas",9,"bold"), bg=bg, fg=fg,
                  relief="flat", cursor="hand2", padx=8, pady=7,
                  activebackground=PANEL2, activeforeground=fg
                  ).pack(fill="x", padx=12, pady=3)

    def _slider(self, parent, label, var, lo, hi, suffix=""):
        f = tk.Frame(parent, bg=PANEL); f.pack(fill="x", padx=12, pady=2)
        lv = tk.StringVar(value=f"{label}: {var.get()}{suffix}")
        tk.Label(f, textvariable=lv, font=("Consolas",8), fg=FG, bg=PANEL).pack(anchor="w")
        def on(val): lv.set(f"{label}: {val}{suffix}"); self.process()
        tk.Scale(f, from_=lo, to=hi, orient="horizontal", variable=var, command=on,
                 bg=PANEL, fg=MUTED, troughcolor=PANEL2, highlightthickness=0,
                 relief="flat", sliderlength=14).pack(fill="x")

    def _check(self, parent, label, var):
        tk.Checkbutton(parent, text=label, variable=var, command=self.process,
                       font=("Consolas",9), fg=FG, bg=PANEL,
                       selectcolor=BG, activebackground=PANEL,
                       relief="flat", cursor="hand2"
                       ).pack(anchor="w", padx=14, pady=2)

    def _sidebar(self, parent):
        self._sec(parent, "INPUT")
        self._btn(parent, "Upload B&W Image",  self.upload,      ACCENT,  BG)
        self._btn(parent, "Load Sample Image", self.load_sample, PANEL2, ACCENT2)

        self._sec(parent, "COLOR STYLE  (10 presets)")
        pf = tk.Frame(parent, bg=PANEL); pf.pack(fill="x", padx=8)
        for name in PRESET_NAMES:
            dot = DOT_COLORS.get(name,"#fff")
            tk.Radiobutton(pf, text=f"  {name}",
                variable=self.preset_var, value=name, command=self.process,
                font=("Consolas",9), fg=dot, bg=PANEL,
                selectcolor=BG, activebackground=PANEL,
                relief="flat", cursor="hand2"
            ).pack(fill="x", padx=6, pady=1)

        self._sec(parent, "COLORIZATION INTENSITY")
        self._slider(parent, "Intensity", self.intensity, 0, 100, "%")

        self._sec(parent, "ACTIONS")
        self._btn(parent, "Compare B&W vs Color", self.toggle_compare, PANEL2, ACCENT2)
        self.cmp_label = tk.StringVar(value="Compare: OFF")
        tk.Label(parent, textvariable=self.cmp_label,
                 font=("Consolas",8), fg=MUTED, bg=PANEL).pack()
        self._btn(parent, "Save Colorized",     self.save_single,     SUCCESS, "white")
        self._btn(parent, "Save Side-by-Side",  self.save_comparison, PANEL2,  SUCCESS)
        self._btn(parent, "Undo",               self.undo,            PANEL2,  MUTED)
        self._btn(parent, "Reset All",          self.reset,           PANEL2,  MUTED)

    def _center(self, parent):
        tk.Label(parent, text="Drag the orange line to compare  ·  Scroll = zoom",
                 font=("Consolas",8), fg=MUTED, bg=BG).pack(pady=(0,4))

        self.canvas = tk.Canvas(parent, bg=PANEL2, highlightthickness=1,
                                highlightbackground=PANEL2, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>",       self._on_resize)
        self.canvas.bind("<ButtonPress-1>",   self._drag_start)
        self.canvas.bind("<B1-Motion>",       self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.canvas.bind("<MouseWheel>",      self._zoom)

        zf = tk.Frame(parent, bg=BG); zf.pack(fill="x", pady=4)
        self.zoom_var = tk.DoubleVar(value=1.0)
        tk.Label(zf, text="Zoom", font=("Consolas",8), fg=MUTED, bg=BG).pack(side="left", padx=8)
        tk.Scale(zf, from_=0.3, to=3.0, resolution=0.1, orient="horizontal",
                 variable=self.zoom_var, command=lambda _: self._redraw(),
                 bg=BG, fg=MUTED, troughcolor=PANEL, highlightthickness=0,
                 relief="flat", length=160).pack(side="left")

    def _right(self, parent):
        def sec2(t):
            tk.Frame(parent, bg=PANEL2, height=1).pack(fill="x", padx=10, pady=(14,0))
            tk.Label(parent, text=t, font=("Consolas",8,"bold"),
                     fg=ACCENT2, bg=PANEL, anchor="w"
                     ).pack(fill="x", padx=14, pady=(6,4))

        sec2("IMAGE ADJUSTMENTS")
        self._slider(parent, "Brightness", self.brightness, 50, 200, "%")
        self._slider(parent, "Contrast",   self.contrast,   50, 200, "%")
        self._slider(parent, "Saturation", self.saturation,  0, 300, "%")
        self._slider(parent, "Sharpness",  self.sharpness,  50, 200, "%")

        sec2("FILTERS")
        self._check(parent, "Auto Enhance (Equalize)", self.auto_enhance)
        self._check(parent, "Denoise",                  self.denoise)

        sec2("HISTOGRAM  (brightness)")
        self.hist_canvas = tk.Canvas(parent, bg=PANEL2, height=80, highlightthickness=0)
        self.hist_canvas.pack(fill="x", padx=10, pady=4)

        sec2("STATS")
        self.stat_labels = {}
        for k in ["Size","Preset","Intensity","Status"]:
            row = tk.Frame(parent, bg=PANEL); row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=k+":", font=("Consolas",8), fg=MUTED, bg=PANEL,
                     width=9, anchor="w").pack(side="left")
            v = tk.StringVar(value="—"); self.stat_labels[k] = v
            tk.Label(row, textvariable=v, font=("Consolas",8,"bold"),
                     fg=FG, bg=PANEL, anchor="w").pack(side="left")

    # ── Actions ───────────────────────────────────────────────────
    def upload(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),("All","*.*")])
        if not path: return
        img = cv2.imread(path)
        if img is None: messagebox.showerror("Error","Cannot read image."); return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.orig_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.undo_stack.clear()
        h,w = img.shape[:2]
        self.stat_labels["Size"].set(f"{w}×{h}")
        self.status.set(f"Loaded: {os.path.basename(path)}")
        self.process()

    def load_sample(self):
        img = np.zeros((400,600), dtype=np.uint8)
        for y in range(200): img[y,:] = int(200+y*0.25)
        img[200:,:] = 80
        cv2.circle(img,(480,80),50,240,-1)
        pts = np.array([[0,200],[150,140],[300,160],[450,130],[600,200]],np.int32)
        cv2.fillPoly(img,[pts],110)
        for tx in [100,220,380,500]:
            cv2.rectangle(img,(tx-8,200),(tx+8,280),50,-1)
            cv2.circle(img,(tx,185),35,60,-1)
        self.orig_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.undo_stack.clear()
        self.stat_labels["Size"].set("600×400")
        self.status.set("Sample landscape loaded.")
        self.process()

    def process(self, *_):
        if self.orig_bgr is None: return
        if self.proc_bgr is not None:
            self.undo_stack.append(self.proc_bgr.copy())
            if len(self.undo_stack) > 10: self.undo_stack.pop(0)

        preset = self.preset_var.get()
        self.proc_bgr = colorize(
            self.orig_bgr, preset,
            self.intensity.get(), self.brightness.get(),
            self.contrast.get(),  self.saturation.get(),
            self.sharpness.get(), self.denoise.get(),
            self.auto_enhance.get())

        self.stat_labels["Preset"].set(preset[:14])
        self.stat_labels["Intensity"].set(f"{self.intensity.get()}%")
        self.stat_labels["Status"].set("Done ✓")
        self.status.set(f"Colorized — preset: {preset}")
        self._update_hist()
        self._redraw()

    def _redraw(self):
        if self.orig_bgr is None: return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return

        if self.compare_on and self.proc_bgr is not None:
            # Initialise divider at centre first time
            if self.div_x is None or self.div_x <= 0:
                self.div_x = cw // 2
            arr = render_compare(self.orig_bgr, self.proc_bgr, cw, ch, self.div_x)
            pil   = Image.fromarray(arr)
            photo = ImageTk.PhotoImage(pil)
        else:
            show  = self.proc_bgr if self.proc_bgr is not None else self.orig_bgr
            zoom  = self.zoom_var.get()
            tw,th = int(cw*zoom), int(ch*zoom)
            r     = fit(show, tw, th)
            rh,rw = r.shape[:2]
            canvas_img = np.full((ch,cw,3),10,dtype=np.uint8)
            ox=(cw-rw)//2; oy=(ch-rh)//2
            canvas_img[oy:oy+rh, ox:ox+rw] = cv2.cvtColor(r,cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(canvas_img))

        self.canvas.delete("all")
        self.canvas.create_image(0,0,anchor="nw",image=photo)
        self.canvas._photo = photo

    def _update_hist(self):
        if self.orig_bgr is None: return
        gray = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2GRAY)
        hist_img = make_histogram(gray, width=206, height=75)
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(hist_img,cv2.COLOR_BGR2RGB)))
        self.hist_canvas.delete("all")
        self.hist_canvas.create_image(0,0,anchor="nw",image=photo)
        self.hist_canvas._photo = photo

    # ── Compare drag ──────────────────────────────────────────────
    def toggle_compare(self):
        self.compare_on = not self.compare_on
        cw = self.canvas.winfo_width()
        self.div_x = cw // 2   # reset to centre every toggle
        lbl = "ON  ← drag orange line →" if self.compare_on else "OFF"
        self.cmp_label.set(f"Compare: {lbl}")
        self.status.set("Drag the orange divider left/right to compare." if self.compare_on else "Compare mode off.")
        self._redraw()

    def _drag_start(self, e):
        if self.compare_on: self.dragging = True

    def _drag_move(self, e):
        if self.dragging and self.compare_on:
            self.div_x = e.x
            self._redraw()

    def _drag_end(self, e):
        self.dragging = False

    def _on_resize(self, e):
        if self.div_x is None:
            self.div_x = e.width // 2
        self._redraw()

    def _zoom(self, e):
        delta = 0.1 if e.delta > 0 else -0.1
        new = max(0.3, min(3.0, self.zoom_var.get() + delta))
        self.zoom_var.set(new)
        self._redraw()

    # ── Save ──────────────────────────────────────────────────────
    def save_single(self):
        if self.proc_bgr is None:
            messagebox.showwarning("Nothing to save","Process an image first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg"),("PNG","*.png")],
            initialfile="colorized_output.jpg")
        if path:
            cv2.imwrite(path, self.proc_bgr)
            messagebox.showinfo("Saved ✅", f"Saved to:\n{path}")
            self.status.set(f"Saved: {path}")

    def save_comparison(self):
        if self.orig_bgr is None or self.proc_bgr is None:
            messagebox.showwarning("Nothing to save","Process an image first."); return
        h = max(self.orig_bgr.shape[0], self.proc_bgr.shape[0])
        o = cv2.resize(self.orig_bgr,(self.orig_bgr.shape[1],h))
        p = cv2.resize(self.proc_bgr,(self.proc_bgr.shape[1],h))
        div = np.full((h,6,3),[22,115,249],dtype=np.uint8)
        comp = np.hstack([o,div,p])
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg"),("PNG","*.png")],
            initialfile="comparison_output.jpg")
        if path:
            cv2.imwrite(path,comp)
            messagebox.showinfo("Saved ✅",f"Side-by-side saved:\n{path}")
            self.status.set(f"Comparison saved: {path}")

    def undo(self):
        if not self.undo_stack: self.status.set("Nothing to undo."); return
        self.proc_bgr = self.undo_stack.pop()
        self._redraw(); self.status.set("Undo applied.")

    def reset(self):
        self.orig_bgr=None; self.proc_bgr=None
        self.undo_stack.clear(); self.compare_on=False; self.div_x=None
        self.canvas.delete("all"); self.hist_canvas.delete("all")
        self.preset_var.set(PRESET_NAMES[0])
        self.intensity.set(85);  self.brightness.set(100)
        self.contrast.set(105);  self.saturation.set(120)
        self.sharpness.set(110); self.denoise.set(False)
        self.auto_enhance.set(False); self.cmp_label.set("Compare: OFF")
        for k in self.stat_labels: self.stat_labels[k].set("—")
        self.status.set("Reset — upload a B&W image to begin.")

# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()
