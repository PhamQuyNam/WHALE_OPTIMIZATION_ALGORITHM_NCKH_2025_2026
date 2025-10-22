import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os
import pickle


class MovieLensDataProcessor:
    """
    Lớp xử lý và chuẩn bị dữ liệu MovieLens, bao gồm Temporal Split.
    """

    def __init__(self, ratings_file='ratings.dat', movies_file='movies.dat',
                 min_user_interactions=5, min_item_interactions=5, test_ratio=0.2):

        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.MIN_USER_INTERACTIONS = min_user_interactions
        self.MIN_ITEM_INTERACTIONS = min_item_interactions
        self.TEST_RATIO = test_ratio  # Tỷ lệ dữ liệu test (ví dụ: 0.2 cho 20%)

        # DataFrames
        self.ratings_df = None
        self.movies_df = None
        self.ratings_df_filtered = None  # DataFrame đã lọc, chứa cả u_idx và i_idx

        # Ma trận và ánh xạ
        self.sparse_matrix_train = None
        self.sparse_matrix_test = None
        self.item_feature_matrix = None
        self.user_map = None
        self.item_map = None

    def _load_data(self):
        """Tải dữ liệu thô từ file .dat và giữ lại timestamp."""
        print("--- 1. Tải Dữ Liệu Thô ---")

        # Tải Ratings (GIỮ LẠI TIMESTAMP)
        self.ratings_df = pd.read_csv(
            self.ratings_file, sep='::', engine='python', header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )  # Giữ tất cả 4 cột ban đầu

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
        print("Mã hóa hoàn tất.")

    def _temporal_split(self):
        """
        Chia dữ liệu thành Train và Test dựa trên timestamp (Temporal Split).
        Train: Tương tác cũ hơn. Test: Tương tác mới hơn.
        """
        print(f"--- 4. Temporal Split (Tỉ lệ Test: {self.TEST_RATIO * 100}%) ---")

        # 1. Sắp xếp theo User và sau đó theo Timestamp
        df = self.ratings_df_filtered.sort_values(by=['u_idx', 'timestamp']).reset_index(drop=True)

        # 2. Đánh dấu các tương tác cho tập Test (Lấy X% tương tác mới nhất của mỗi User)
        df['rank'] = df.groupby('u_idx')['timestamp'].rank(method='first', ascending=False)

        # Tính ngưỡng chia
        df['count'] = df.groupby('u_idx')['u_idx'].transform('count')
        test_size_per_user = (df['count'] * self.TEST_RATIO).apply(np.ceil)

        # Nếu rank <= test_size_per_user, đó là tương tác mới nhất và đưa vào test
        df_test = df[df['rank'] <= test_size_per_user]
        df_train = df[df['rank'] > test_size_per_user]

        # Đảm bảo mỗi user có ít nhất 1 tương tác trong tập train
        # Lọc những users bị đẩy toàn bộ tương tác vào test (nếu có)
        users_in_train = df_train['u_idx'].unique()
        df_test = df_test[df_test['u_idx'].isin(users_in_train)]

        print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")

        return df_train, df_test

    def _create_interaction_matrix(self, df_train, df_test):
        """Tạo Ma trận tương tác Train và Test thưa thớt."""
        print("--- 5. Tạo Ma Trận Tương Tác Thưa Thớt ---")
        num_users = self.ratings_df_filtered['u_idx'].nunique()
        num_items = self.ratings_df_filtered['i_idx'].nunique()

        # Tạo Sparse Matrix cho tập Train
        self.sparse_matrix_train = csr_matrix((
            df_train['rating'].values,
            (df_train['u_idx'].values, df_train['i_idx'].values)
        ), shape=(num_users, num_items))

        # Tạo Sparse Matrix cho tập Test
        self.sparse_matrix_test = csr_matrix((
            df_test['rating'].values,
            (df_test['u_idx'].values, df_test['i_idx'].values)
        ), shape=(num_users, num_items))

        # Dữ liệu train matrix (dày) - Chỉ cần cho khởi tạo ban đầu, không cần lưu
        self.user_item_matrix = self.sparse_matrix_train.toarray()

        print(f"Kích thước Train Matrix: {self.sparse_matrix_train.shape}")
        print(f"Kích thước Test Matrix: {self.sparse_matrix_test.shape}")

    def _create_item_feature_matrix(self):
        """Tạo Item Feature Matrix (Multi-Hot Encoding cho Genres)."""
        print("--- 6. Tạo Ma Trận Thuộc Tính (Genre) ---")
        movies_df_filtered = self.movies_df[
            self.movies_df['item_id'].isin(self.ratings_df_filtered['item_id'].unique())].copy()

        movies_df_filtered['i_idx'] = movies_df_filtered['item_id'].map(self.item_map)
        movies_df_filtered = movies_df_filtered.sort_values(by='i_idx').reset_index(drop=True)

        genres = movies_df_filtered['genres'].str.get_dummies('|')
        self.item_feature_matrix = genres.values

        print(f"Kích thước Ma trận Thuộc tính: {self.item_feature_matrix.shape}")

    def run_pipeline(self):
        """Chạy toàn bộ quá trình xử lý dữ liệu."""
        self._load_data()
        self._filter_interactions()
        self._encode_ids()

        # Chạy Temporal Split và tạo ma trận
        df_train, df_test = self._temporal_split()
        self._create_interaction_matrix(df_train, df_test)

        self._create_item_feature_matrix()

        return {
            'sparse_matrix_train': self.sparse_matrix_train,
            'sparse_matrix_test': self.sparse_matrix_test,
            'item_feature_matrix': self.item_feature_matrix,
            'user_map': self.user_map,
            'item_map': self.item_map
        }

    def save_processed_data(self, output_dir='processed_data'):
        """Lưu các ma trận và ánh xạ ID đã xử lý ra file."""
        if self.sparse_matrix_train is None or self.item_feature_matrix is None:
            print("Lỗi: Dữ liệu chưa được xử lý. Vui lòng chạy run_pipeline() trước.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"--- Lưu dữ liệu đã xử lý vào thư mục '{output_dir}' ---")

        # 1. Lưu Ma trận tương tác Train và Test
        save_npz(os.path.join(output_dir, 'sparse_matrix_train.npz'), self.sparse_matrix_train)
        save_npz(os.path.join(output_dir, 'sparse_matrix_test.npz'), self.sparse_matrix_test)
        print("Đã lưu sparse_matrix_train.npz và sparse_matrix_test.npz")

        # 2. Lưu Ma trận thuộc tính Item (NumPy array)
        np.save(os.path.join(output_dir, 'item_feature_matrix.npy'), self.item_feature_matrix)
        print("Đã lưu item_feature_matrix.npy")

        # 3. Lưu Ánh xạ ID (Maps)
        with open(os.path.join(output_dir, 'user_map.pkl'), 'wb') as f:
            pickle.dump(self.user_map, f)
        with open(os.path.join(output_dir, 'item_map.pkl'), 'wb') as f:
            pickle.dump(self.item_map, f)
        print("Đã lưu user_map.pkl và item_map.pkl")

        print("\n--- HOÀN TẤT LƯU TRỮ ---")


# --- CÁCH SỬ DỤNG VÀ CHẠY CHƯƠNG TRÌNH ---
if __name__ == '__main__':
    # 1. Khởi tạo đối tượng xử lý (Có thể tùy chỉnh test_ratio)
    # Ví dụ: Tỷ lệ 80/20
    processor = MovieLensDataProcessor(test_ratio=0.2)

    # 2. Chạy toàn bộ pipeline xử lý
    data_output = processor.run_pipeline()

    # 3. GỌI PHƯƠNG THỨC LƯU DỮ LIỆU ĐÃ XỬ LÝ
    processor.save_processed_data(output_dir='processed_data_split')

    print("\nKiểm tra thư mục 'processed_data_split' để xem kết quả.")