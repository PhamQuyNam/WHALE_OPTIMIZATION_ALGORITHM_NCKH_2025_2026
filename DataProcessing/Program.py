import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class MovieLensDataProcessor:
    """
    Lớp xử lý và chuẩn bị dữ liệu MovieLens cho mô hình Recommender System.
    Đầu ra chính là User-Item Matrix và Item Feature Matrix.
    """

    def __init__(self, ratings_file='ratings.dat', movies_file='movies.dat',
                 min_user_interactions=5, min_item_interactions=5):
        # 1. Khởi tạo các tham số cơ bản
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.MIN_USER_INTERACTIONS = min_user_interactions
        self.MIN_ITEM_INTERACTIONS = min_item_interactions

        # 2. Các DataFrame/Matrix sẽ được tạo ra
        self.ratings_df = None
        self.movies_df = None
        self.ratings_df_filtered = None
        self.user_item_matrix = None
        self.sparse_matrix = None
        self.item_feature_matrix = None

        # 3. Ánh xạ ID
        self.user_map = None
        self.item_map = None

    def _load_data(self):
        """Tải dữ liệu thô từ file .dat."""
        print("--- 1. Tải Dữ Liệu Thô ---")

        # Tải Ratings
        self.ratings_df = pd.read_csv(
            self.ratings_file, sep='::', engine='python', header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )[['user_id', 'item_id', 'rating']]

        # Tải Movies
        self.movies_df = pd.read_csv(
            self.movies_file, sep='::', engine='python', header=None,
            names=['item_id', 'title', 'genres'], encoding='latin-1'
        )
        print("Tải dữ liệu Ratings và Movies thành công.")

    def _filter_interactions(self):
        """Lọc bỏ Users và Items có tương tác thấp."""
        print("--- 2. Lọc Tương Tác Thấp ---")
        ratings_df = self.ratings_df.copy()

        # Lọc Items
        item_counts = ratings_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.MIN_ITEM_INTERACTIONS].index
        ratings_df = ratings_df[ratings_df['item_id'].isin(valid_items)]

        # Lọc Users
        user_counts = ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.MIN_USER_INTERACTIONS].index
        self.ratings_df_filtered = ratings_df[ratings_df['user_id'].isin(valid_users)]

        print(
            f"Users/Items sau lọc: {self.ratings_df_filtered['user_id'].nunique()} Users, {self.ratings_df_filtered['item_id'].nunique()} Items.")

    def _encode_ids(self):
        """Mã hóa ID gốc thành ID liên tục (0, 1, 2...)."""
        print("--- 3. Mã Hóa ID ---")
        unique_users = self.ratings_df_filtered['user_id'].unique()
        unique_items = self.ratings_df_filtered['item_id'].unique()

        self.user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        self.item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

        self.ratings_df_filtered['u_idx'] = self.ratings_df_filtered['user_id'].map(self.user_map)
        self.ratings_df_filtered['i_idx'] = self.ratings_df_filtered['item_id'].map(self.item_map)
        print("Mã hóa hoàn tất. Dữ liệu sẵn sàng cho ma trận.")

    def _create_interaction_matrix(self):
        """Tạo Ma trận tương tác User-Item."""
        print("--- 4. Tạo Ma Trận Tương Tác ---")
        num_users = self.ratings_df_filtered['u_idx'].nunique()
        num_items = self.ratings_df_filtered['i_idx'].nunique()

        # Ma trận dày (dense matrix) - dễ dùng cho khởi tạo Matrix Factorization
        self.user_item_matrix = self.ratings_df_filtered.pivot(
            index='u_idx', columns='i_idx', values='rating'
        ).fillna(0)

        # Ma trận thưa thớt (sparse matrix) - hiệu quả bộ nhớ
        self.sparse_matrix = csr_matrix((
            self.ratings_df_filtered['rating'].values,
            (self.ratings_df_filtered['u_idx'].values, self.ratings_df_filtered['i_idx'].values)
        ), shape=(num_users, num_items))

        print(f"Kích thước Ma trận (dày/thưa): {self.user_item_matrix.shape}")

    def _create_item_feature_matrix(self):
        """Tạo Item Feature Matrix (Multi-Hot Encoding cho Genres)."""
        print("--- 5. Tạo Ma Trận Thuộc Tính (Genre) ---")
        movies_df_filtered = self.movies_df[
            self.movies_df['item_id'].isin(self.ratings_df_filtered['item_id'].unique())].copy()

        # Ánh xạ item_id gốc sang item_id mới (i_idx)
        movies_df_filtered['i_idx'] = movies_df_filtered['item_id'].map(self.item_map)
        movies_df_filtered = movies_df_filtered.sort_values(by='i_idx').reset_index(drop=True)

        # Multi-Hot Encoding cho Genres
        genres = movies_df_filtered['genres'].str.get_dummies('|')

        # Kết hợp và lấy ma trận feature
        self.item_feature_matrix = genres.values

        print(f"Kích thước Ma trận Thuộc tính: {self.item_feature_matrix.shape}")

    def run_pipeline(self):
        """Chạy toàn bộ quá trình xử lý dữ liệu."""
        self._load_data()
        self._filter_interactions()
        self._encode_ids()
        self._create_interaction_matrix()
        self._create_item_feature_matrix()

        return {
            'user_item_matrix': self.user_item_matrix,
            'sparse_matrix': self.sparse_matrix,
            'item_feature_matrix': self.item_feature_matrix,
            'user_map': self.user_map,
            'item_map': self.item_map
        }

    def save_processed_data(self, output_dir='processed_data'):
        """Lưu các ma trận và ánh xạ ID đã xử lý ra file."""
        import os
        import pickle

        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"--- Lưu dữ liệu đã xử lý vào thư mục '{output_dir}' ---")

        # 1. Lưu Ma trận tương tác (Sparse Matrix - Hiệu quả nhất)
        # Sử dụng định dạng .npz cho ma trận thưa thớt
        from scipy.sparse import save_npz
        save_npz(os.path.join(output_dir, 'sparse_matrix.npz'), self.sparse_matrix)
        print("Đã lưu sparse_matrix.npz")

        # 2. Lưu Ma trận thuộc tính Item (NumPy array)
        np.save(os.path.join(output_dir, 'item_feature_matrix.npy'), self.item_feature_matrix)
        print("Đã lưu item_feature_matrix.npy")

        # 3. Lưu Ánh xạ ID (Maps)
        # Sử dụng pickle để lưu dictionary
        with open(os.path.join(output_dir, 'user_map.pkl'), 'wb') as f:
            pickle.dump(self.user_map, f)
        with open(os.path.join(output_dir, 'item_map.pkl'), 'wb') as f:
            pickle.dump(self.item_map, f)
        print("Đã lưu user_map.pkl và item_map.pkl")
# --- CÁCH SỬ DỤNG ---
# processor = MovieLensDataProcessor()
# data_output = processor.run_pipeline()

# ma_tran_tuong_tac = data_output['user_item_matrix']
# ma_tran_thuoc_tinh = data_output['item_feature_matrix']

# print("\n--- KẾT QUẢ CUỐI CÙNG ---")
# print(f"Ma trận tương tác (dày) đã sẵn sàng để truyền vào mô hình WOA: \n{ma_tran_tuong_tac.head()}")

# --- CÁCH SỬ DỤNG VÀ CHẠY CHƯƠNG TRÌNH ---

# 1. Khởi tạo đối tượng xử lý
processor = MovieLensDataProcessor()

# 2. Chạy toàn bộ pipeline xử lý (Đảm bảo dữ liệu được tạo ra)
data_output = processor.run_pipeline()

# 3. GỌI PHƯƠNG THỨC LƯU DỮ LIỆU ĐÃ XỬ LÝ (Đây là bước bị thiếu/comment)
processor.save_processed_data(output_dir='processed_data')

print("\n--- HOÀN TẤT ---")
print("Vui lòng kiểm tra thư mục 'processed_data' trong thư mục hiện tại.")