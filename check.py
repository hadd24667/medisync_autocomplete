import re
import pandas as pd

def normalize_forms_vn(s):
    """
    Chuẩn hóa dạng thuốc tiếng Việt:
    - Tách theo dấu ;
    - Ví dụ: "Viên nén 125 mg, 250 mg"
      -> ["Viên nén 125 mg", "Viên nén 250 mg"]
    - Giữ nguyên dạng đơn
    - Không xóa dấu thập phân (0.3%)
    - Bỏ item rỗng hoặc không có hàm lượng
    """

    if pd.isna(s):
        return []

    # Chỉ xoá dấu chấm ở cuối câu, không xoá 0.3%
    text = re.sub(r"\.$", "", str(s).strip())

    # Tách theo dấu ;
    parts = [p.strip() for p in text.split(";") if p.strip()]

    result = []

    for p in parts:
        # CASE 1: có dấu phẩy → nhiều hàm lượng
        if "," in p:
            items = [i.strip() for i in p.split(",") if i.strip()]

            first = items[0]

            # tìm base = phần trước số đầu tiên
            m = re.match(r"^(.*?)(\d+.*)$", first)
            if m:
                base = m.group(1).strip()
            else:
                tokens = first.split()
                idx = next((i for i, t in enumerate(tokens) if any(c.isdigit() for c in t)), None)
                base = " ".join(tokens[:idx]) if idx else first

            # Tạo từng item
            for it in items:
                if not re.search(r"\d", it):
                    continue  # Skip item không có số → tránh "Viên nén"

                qty = re.findall(r"\d+.*", it)
                if qty:
                    result.append(f"{base} {qty[0]}".strip())

        else:
            # CASE 2: dạng đơn
            if p.strip():
                result.append(p.strip())

    return result


if __name__ == "__main__":
    # Ví dụ sử dụng hàm
   test_str_list = [
    # 1 — nhiều hàm lượng chung dạng thuốc
    "Viên nén 50 mg, 100 mg, 250 mg",

    # 2 — nhiều dạng + nhiều hàm lượng
    "Viên nén 125 mg, 250 mg; Thuốc tiêm 500 mg/5 ml",

    # 3 — dạng bị lặp khi copy từ PDF
    "Viên nén Viên nén 200 mg, 400 mg; Thuốc uống 10 mg",

    # 4 — dạng cốm bị lặp
    "Thuốc cốm Thuốc cốm 100 mg, 200 mg",

    # 5 — nhiều dạng khác nhau trong 1 dòng
    "Dung dịch 3%; Bột tinh thể; Viên nang 250 mg",

    # 6 — đơn vị lạ: IU, %, gói
    "Thuốc nhỏ mắt 0.3%; Gói bột 1 g; Dung dịch 5000 IU",

    # 7 — dạng tiêm có đơn vị hỗn hợp
    "Thuốc tiêm 100 mg/2 ml, 200 mg/4 ml",

    # 8 — dạng sủi, dạng xịt, đơn vị biến thể
    "Viên sủi 600 mg, 800 mg; Xịt mũi 0.1%",

    # 9 — dạng hỗn hợp phức tạp + nhiều dấu phẩy
    "Viên nén 5 mg, 10 mg, 20 mg; Thuốc uống 100 mg",

    # 10 — test dạng không có hàm lượng sau dấu phẩy (edge-case)
    "Viên nén 500 mg, ; Thuốc tiêm 1 g"
    ]
for s in test_str_list:
         print(f"Input: {s}")
         forms = normalize_forms_vn(s)
         print("Output:")
         for f in forms:
              print(f" - {f}")
         print()
