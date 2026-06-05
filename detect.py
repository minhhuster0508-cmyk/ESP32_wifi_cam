import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def fit_display(img, max_w=1000, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1:
        img2 = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img2 = img.copy()
    return img2


def align_orb(ref, test):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)

    if des1 is None or des2 is None:
        return None, "Không tìm được đủ đặc trưng ORB."

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 20:
        return None, "Không đủ điểm match để căn chỉnh ảnh."

    good = matches[:max(30, int(len(matches) * 0.25))]

    ref_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    test_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(test_pts, ref_pts, cv2.RANSAC, 5.0)

    if H is None:
        return None, "Không tính được Homography."

    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(test, H, (w, h))

    return aligned, "OK"


def detect_missing_auto(ref, test_aligned, min_area=120, threshold_value=45):
    ref_blur = cv2.GaussianBlur(ref, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_aligned, (5, 5), 0)

    diff = cv2.absdiff(ref_blur, test_blur)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(diff_gray, threshold_value, 255, cv2.THRESH_BINARY)

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.dilate(mask, kernel1, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = test_aligned.copy()
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 8 or h < 8:
            continue

        roi_mask = mask[y:y+h, x:x+w]
        changed_ratio = cv2.countNonZero(roi_mask) / (w * h)

        if changed_ratio < 0.08:
            continue

        boxes.append((x, y, w, h, area, changed_ratio))

    boxes = merge_boxes(boxes)

    for i, (x, y, w, h, area, ratio) in enumerate(boxes, start=1):
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            result,
            f"NGHI THIEU {i}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    return result, mask, boxes


def merge_boxes(boxes, gap=12):
    if not boxes:
        return []

    rects = []
    for x, y, w, h, area, ratio in boxes:
        rects.append([x, y, x + w, y + h, area, ratio])

    merged = True

    while merged:
        merged = False
        new_rects = []
        used = [False] * len(rects)

        for i in range(len(rects)):
            if used[i]:
                continue

            x1, y1, x2, y2, area1, ratio1 = rects[i]
            used[i] = True

            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue

                a1, b1, a2, b2, area2, ratio2 = rects[j]

                overlap = not (
                    x2 + gap < a1 or
                    a2 + gap < x1 or
                    y2 + gap < b1 or
                    b2 + gap < y1
                )

                if overlap:
                    x1 = min(x1, a1)
                    y1 = min(y1, b1)
                    x2 = max(x2, a2)
                    y2 = max(y2, b2)
                    area1 += area2
                    ratio1 = max(ratio1, ratio2)
                    used[j] = True
                    merged = True

            new_rects.append([x1, y1, x2, y2, area1, ratio1])

        rects = new_rects

    out = []
    for x1, y1, x2, y2, area, ratio in rects:
        out.append((x1, y1, x2 - x1, y2 - y1, area, ratio))

    return out


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto PCB Missing Component Detector")

        self.ref_img = None
        self.test_img = None
        self.aligned_img = None
        self.result_img = None
        self.mask_img = None

        self.min_area = tk.IntVar(value=120)
        self.thresh = tk.IntVar(value=45)

        self.build_ui()

    def build_ui(self):
        left = tk.Frame(self.root, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        tk.Button(left, text="1. Chọn ảnh mạch chuẩn OK", command=self.load_ref).pack(fill=tk.X, pady=4)
        tk.Button(left, text="2. Chọn ảnh mạch cần kiểm tra", command=self.load_test).pack(fill=tk.X, pady=4)
        tk.Button(left, text="3. Auto detect thiếu linh kiện", command=self.run_detect).pack(fill=tk.X, pady=8)

        tk.Label(left, text="Ngưỡng khác biệt pixel").pack(anchor="w", pady=(12, 0))
        tk.Scale(left, from_=10, to=100, orient=tk.HORIZONTAL, variable=self.thresh).pack(fill=tk.X)

        tk.Label(left, text="Diện tích lỗi nhỏ nhất").pack(anchor="w", pady=(12, 0))
        tk.Scale(left, from_=20, to=2000, orient=tk.HORIZONTAL, variable=self.min_area).pack(fill=tk.X)

        tk.Button(left, text="Xem ảnh chuẩn", command=self.show_ref).pack(fill=tk.X, pady=(16, 4))
        tk.Button(left, text="Xem ảnh test đã căn chỉnh", command=self.show_aligned).pack(fill=tk.X, pady=4)
        tk.Button(left, text="Xem mask lỗi", command=self.show_mask).pack(fill=tk.X, pady=4)
        tk.Button(left, text="Xem kết quả", command=self.show_result).pack(fill=tk.X, pady=4)
        tk.Button(left, text="Lưu ảnh kết quả", command=self.save_result).pack(fill=tk.X, pady=12)

        tk.Label(left, text="Log:").pack(anchor="w")
        self.log = tk.Text(left, height=18, width=40)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(right, bg="#222222")
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def log_write(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def show_cv_image(self, img):
        if img is None:
            return

        disp = fit_display(img)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil)

        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

    def load_ref(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh mạch chuẩn",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not path:
            return

        img = imread_unicode(path)

        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh chuẩn.")
            return

        self.ref_img = img
        self.show_cv_image(img)
        self.log_write(f"Đã load ảnh chuẩn: {Path(path).name}")

    def load_test(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh mạch cần kiểm tra",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not path:
            return

        img = imread_unicode(path)

        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh test.")
            return

        self.test_img = img
        self.show_cv_image(img)
        self.log_write(f"Đã load ảnh test: {Path(path).name}")

    def run_detect(self):
        if self.ref_img is None:
            messagebox.showwarning("Thiếu ảnh chuẩn", "Chọn ảnh mạch chuẩn trước.")
            return

        if self.test_img is None:
            messagebox.showwarning("Thiếu ảnh test", "Chọn ảnh test trước.")
            return

        self.log_write("Đang tự căn chỉnh ảnh bằng ORB + Homography...")

        aligned, msg = align_orb(self.ref_img, self.test_img)

        if aligned is None:
            self.log_write("Lỗi align: " + msg)
            messagebox.showerror("Lỗi align", msg)
            return

        self.aligned_img = aligned
        self.log_write("Căn chỉnh ảnh: OK")

        self.log_write("Đang so ảnh chuẩn với ảnh test...")

        result, mask, boxes = detect_missing_auto(
            self.ref_img,
            self.aligned_img,
            min_area=self.min_area.get(),
            threshold_value=self.thresh.get()
        )

        self.result_img = result
        self.mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        self.show_cv_image(result)

        if len(boxes) == 0:
            self.log_write("KẾT LUẬN: OK hoặc chưa phát hiện vùng thiếu rõ ràng.")
        else:
            self.log_write(f"KẾT LUẬN: phát hiện {len(boxes)} vùng nghi thiếu linh kiện.")
            for i, (x, y, w, h, area, ratio) in enumerate(boxes, start=1):
                self.log_write(
                    f"Vùng {i}: x={x}, y={y}, w={w}, h={h}, area={area:.1f}, changed={ratio:.2f}"
                )

    def show_ref(self):
        self.show_cv_image(self.ref_img)

    def show_aligned(self):
        self.show_cv_image(self.aligned_img)

    def show_mask(self):
        self.show_cv_image(self.mask_img)

    def show_result(self):
        self.show_cv_image(self.result_img)

    def save_result(self):
        if self.result_img is None:
            messagebox.showwarning("Chưa có kết quả", "Chạy detect trước.")
            return

        path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
        )

        if not path:
            return

        ext = Path(path).suffix
        ok, buf = cv2.imencode(ext, self.result_img)

        if ok:
            buf.tofile(path)
            self.log_write(f"Đã lưu ảnh kết quả: {path}")
        else:
            messagebox.showerror("Lỗi", "Không lưu được ảnh.")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1350x820")
    app = App(root)
    root.mainloop()