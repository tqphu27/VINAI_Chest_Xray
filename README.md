# VINAI_Chest_Xray

Project được chia thành 2 phần chính:
- Xử lý dữ liệu (Gồm 14 class).
- Huấn luyện model bằng cách sử dụng mô hình yolo5.
  
 1. Xử lý dữ liệu 
- Dữ liệu cuộc thi khá lớn do ở dạng DICOM nên ở đây mình sử dụng một bộ dataset đã được convert sang PNG. [Here](https://www.miai.vn/thu-vien-mi-ai/.)
- Dữ liệu gồm 3 file:
  +   Một file csv- chứa thông tin về ảnh, nhãn của các vùng chuẩn đoán.
  +   Một folder train: chứa các ảnh để train model.
  +   Một foler test: Chứa các ảnh để test model.
- Tiền xử lý dữ liệu:(preprocess.py)
  + Đầu tiên ta sẽ sử dụng file csv để lấy tên file và tọa độ các box, nhãn của các bệnh chuẩn đoán.
  + Tiếp tục ta sẽ đọc file ảnh để biết kích thước ảnh.
  + Convert sang YOLO format và ghi vào folder train.
- Chia dữ liệu thành train và val theo cấu trúc YoloV5 (split.py)

2. Huấn luyện mô hình 
- Chọn mô hình để huấn luyện(Ở đây mình chọn 2 mô hình yolo5m và yolo5x).
- Trong mỗi mô hình, vào file yaml tương ứng và sửa thông số nc=14(trùng với số class)
- Tạo một file data.yml trong models: 
- <img src="https://user-images.githubusercontent.com/90370260/155838525-b26c0749-67a8-44a9-bc4d-e570a404e0a5.png" alt="200" width="300" />
- Sau đó training
  + Theo model yolo5m: [Notebook](https://colab.research.google.com/drive/1axv9C87HVGcVCnFdxPSfeZ5TAw6JU63q?authuser=2#scrollTo=yeu77eYsOOCc)
  + Theo model yolo5x: [Notebook](https://colab.research.google.com/drive/1cXx2pt9JLaXPuN40WDBOZl2fxf5arU9T?authuser=4&hl=vi)

# Kết quả cuối cùng được lưu trong đường dẫn sau:
 - Mô hình Yolo5m: [Drive](https://drive.google.com/drive/folders/1cwxqKLQl_a9UINxOQn3RGI_ip0JSmYvl?usp=sharing)
 - Mô hình Yolo5x: [Drive](https://drive.google.com/drive/u/4/folders/12q2rYtIYcN4FJEEtUDnl6i6jrXzTbG_V)
- Nhận xét: mAP của cả hai mô hình có giá trị xấp xỉ nhau(khoảng 0.245), không được cao cho lắm :(
