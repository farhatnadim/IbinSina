"""Tests for data_loading/dataset.py - Dataset classes and data loading."""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from data_loading.dataset import (
    MILDataset,
    GroupedMILDataset,
    HierarchicalMILDataset,
    load_labels,
    get_available_features,
    join_labels_and_features,
    SlideData,
    GroupedData,
    HierarchicalData,
)


class TestLoadLabels:
    """Tests for load_labels function."""

    def test_load_labels_basic(self, sample_labels_csv):
        """Test basic CSV loading."""
        df = load_labels(sample_labels_csv)

        assert 'slide_id' in df.columns
        assert 'label' in df.columns
        assert len(df) == 8  # Updated fixture has 8 slides

    def test_load_labels_string_conversion(self, sample_labels_csv):
        """Test that slide_id and label are converted to strings."""
        df = load_labels(sample_labels_csv)

        assert df['slide_id'].dtype == object  # String type
        assert df['label'].dtype == object

    def test_load_labels_missing_columns(self, temp_dir):
        """Test that missing required columns raise ValueError."""
        csv_path = temp_dir / "bad_labels.csv"
        csv_path.write_text("id,class\na,b\n")

        with pytest.raises(ValueError, match="Missing required columns"):
            load_labels(csv_path)


class TestGetAvailableFeatures:
    """Tests for get_available_features function."""

    def test_get_available_features(self, sample_features_dir):
        """Test getting available feature files."""
        df = get_available_features(sample_features_dir)

        assert 'slide_id' in df.columns
        assert 'h5_path' in df.columns
        assert len(df) == 8  # Updated fixture has 8 slides


class TestJoinLabelsAndFeatures:
    """Tests for join_labels_and_features function."""

    def test_join_labels_and_features(self, sample_labels_csv, sample_features_dir, capsys):
        """Test joining labels with features."""
        labels_df = load_labels(sample_labels_csv)
        features_df = get_available_features(sample_features_dir)

        merged = join_labels_and_features(labels_df, features_df)

        assert len(merged) == 8  # All 8 slides have both labels and features
        assert 'h5_path' in merged.columns

    def test_join_handles_missing_features(self, temp_dir, capsys):
        """Test join when some slides have missing features."""
        # Create labels for 5 slides but only 3 feature files
        csv_path = temp_dir / "labels.csv"
        csv_path.write_text(
            "slide_id,label\n"
            "slide_001,a\n"
            "slide_002,b\n"
            "slide_003,c\n"
            "slide_004,d\n"
            "slide_005,e\n"
        )

        features_dir = temp_dir / "features"
        features_dir.mkdir()
        for i in [1, 2, 3]:  # Only create 3 feature files
            import h5py
            with h5py.File(features_dir / f"slide_00{i}.h5", 'w') as f:
                f.create_dataset('features', data=np.zeros((10, 64)))

        labels_df = load_labels(csv_path)
        features_df = get_available_features(features_dir)
        merged = join_labels_and_features(labels_df, features_df)

        assert len(merged) == 3  # Only matched slides


class TestMILDataset:
    """Tests for MILDataset class."""

    def test_mil_dataset_init(self, sample_labels_csv, sample_features_dir, capsys):
        """Test MILDataset initialization."""
        dataset = MILDataset(sample_labels_csv, sample_features_dir)

        assert len(dataset) == 8  # Updated fixture has 8 slides
        assert dataset.embed_dim == 512

    def test_mil_dataset_getitem_by_index(self, sample_mil_dataset):
        """Test index-based access."""
        slide = sample_mil_dataset[0]

        assert isinstance(slide, SlideData)
        assert isinstance(slide.features, torch.Tensor)
        assert slide.features.dim() == 2
        assert isinstance(slide.label, str)

    def test_mil_dataset_getitem_by_slide_id(self, sample_mil_dataset):
        """Test slide_id-based access."""
        slide = sample_mil_dataset['slide_001']

        assert slide.slide_id == 'slide_001'

    def test_mil_dataset_properties(self, sample_mil_dataset):
        """Test slide_ids and labels properties."""
        assert len(sample_mil_dataset.slide_ids) == 8  # Updated fixture
        assert len(sample_mil_dataset.labels) == 8
        assert 'slide_001' in sample_mil_dataset.slide_ids

    def test_mil_dataset_iteration(self, sample_mil_dataset):
        """Test iteration over dataset."""
        slides = list(sample_mil_dataset)
        assert len(slides) == 8  # Updated fixture
        assert all(isinstance(s, SlideData) for s in slides)

    def test_mil_dataset_random_split(self, sample_mil_dataset):
        """Test random train/val/test split."""
        splits = sample_mil_dataset.random_split(
            train_frac=0.5, val_frac=0.25, seed=42
        )

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == len(sample_mil_dataset)

    def test_mil_dataset_get_subset(self, sample_mil_dataset):
        """Test getting a subset of the dataset."""
        subset = sample_mil_dataset.get_subset(['slide_001', 'slide_002'])

        assert len(subset) == 2
        assert 'slide_001' in subset.slide_ids
        assert 'slide_002' in subset.slide_ids

    def test_mil_dataset_split_by_column(self, sample_mil_dataset):
        """Test splitting by a column."""
        splits = sample_mil_dataset.split_by_column('split')

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

    def test_mil_dataset_create_cv_splits(self, sample_labels_csv_multiclass, sample_features_dir, capsys):
        """Test CV split creation."""
        dataset = MILDataset(sample_labels_csv_multiclass, sample_features_dir)

        test_dataset, folds = dataset.create_cv_splits(
            num_folds=2, test_frac=0.25, seed=42
        )

        assert len(folds) == 2
        assert len(test_dataset) == 2  # 25% of 8 = 2

        for train_ds, val_ds in folds:
            assert len(train_ds) + len(val_ds) == 6  # Remaining slides (8-2=6)

    def test_mil_dataset_load_split_json(self, sample_mil_dataset, sample_split_json):
        """Test loading split from JSON file."""
        train_ds = sample_mil_dataset.load_split(sample_split_json, 'train')

        # Train split has case_1, case_2, case_3, case_7
        # case_1 has 2 slides, others have 1 each = 5 slides total
        assert len(train_ds) == 5


class TestGroupedMILDataset:
    """Tests for GroupedMILDataset class."""

    def test_grouped_dataset_concat_by(self, sample_mil_dataset, capsys):
        """Test early fusion grouping."""
        grouped = sample_mil_dataset.concat_by('case_id')

        assert isinstance(grouped, GroupedMILDataset)
        # Updated fixture has 7 unique cases (case_1 through case_7)
        assert len(grouped) == 7

    def test_grouped_dataset_getitem(self, sample_mil_dataset, capsys):
        """Test accessing grouped data."""
        grouped = sample_mil_dataset.concat_by('case_id')

        group = grouped[0]
        assert isinstance(group, GroupedData)
        assert isinstance(group.features, torch.Tensor)
        assert group.features.dim() == 2

    def test_grouped_dataset_label_voting_first(self, sample_mil_dataset, capsys):
        """Test 'first' label voting."""
        grouped = sample_mil_dataset.concat_by('case_id', label_voting='first')

        # Verify we can iterate and get labels
        for group in grouped:
            assert isinstance(group.label, str)

    def test_grouped_dataset_label_voting_max(self, sample_mil_dataset, capsys):
        """Test 'max' label voting."""
        grouped = sample_mil_dataset.concat_by('case_id', label_voting='max')

        # case_1 has both benign and tumor, 'max' should pick the alphabetically higher one
        case_1 = grouped['case_1']
        assert case_1.label == 'tumor'  # 'tumor' > 'benign' alphabetically

    def test_grouped_dataset_concatenated_features(self, sample_mil_dataset, capsys):
        """Test that features are correctly concatenated."""
        grouped = sample_mil_dataset.concat_by('case_id')

        # case_1 has slide_001 (100 patches) and slide_002 (150 patches) in updated fixture
        case_1 = grouped['case_1']
        # With 8 slides and 7 cases, case_1 has 2 slides (slide_001 + slide_002)
        assert case_1.features.shape[0] == 100 + 150
        assert case_1.num_items == 2

    def test_grouped_dataset_random_split(self, sample_mil_dataset, capsys):
        """Test random split for grouped dataset."""
        grouped = sample_mil_dataset.concat_by('case_id')
        splits = grouped.random_split(train_frac=0.5, val_frac=0.25, seed=42)

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

    def test_grouped_dataset_properties(self, sample_mil_dataset, capsys):
        """Test group_ids and labels properties."""
        grouped = sample_mil_dataset.concat_by('case_id')

        assert len(grouped.group_ids) == 7  # 7 unique cases in updated fixture
        assert len(grouped.labels) == 7


class TestHierarchicalMILDataset:
    """Tests for HierarchicalMILDataset class."""

    def test_hierarchical_dataset_group_by(self, sample_mil_dataset, capsys):
        """Test late fusion grouping."""
        hier = sample_mil_dataset.group_by('case_id')

        assert isinstance(hier, HierarchicalMILDataset)
        assert len(hier) == 7  # 7 unique cases in updated fixture

    def test_hierarchical_dataset_preserves_structure(self, sample_mil_dataset, capsys):
        """Test that hierarchical grouping preserves slide structure."""
        hier = sample_mil_dataset.group_by('case_id')

        # case_1 has 2 slides
        case_1 = hier['case_1']
        assert isinstance(case_1, HierarchicalData)
        assert isinstance(case_1.features, list)
        assert len(case_1.features) == 2

        # Each element should be a separate tensor
        for feat in case_1.features:
            assert isinstance(feat, torch.Tensor)
            assert feat.dim() == 2

    def test_hierarchical_data_to_padded_tensor(self, sample_mil_dataset, capsys):
        """Test converting hierarchical data to padded tensor."""
        hier = sample_mil_dataset.group_by('case_id')
        case_1 = hier['case_1']

        padded, mask = case_1.to_padded_tensor()

        # 2 slides, max patches across them, embed_dim
        assert padded.dim() == 3
        assert padded.shape[0] == 2
        assert mask.shape[0] == 2

    def test_hierarchical_dataset_random_split(self, sample_mil_dataset, capsys):
        """Test random split for hierarchical dataset."""
        hier = sample_mil_dataset.group_by('case_id')
        splits = hier.random_split(train_frac=0.5, val_frac=0.25, seed=42)

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits


class TestDatasetMissingFeatures:
    """Tests for handling missing feature files."""

    def test_dataset_missing_features(self, temp_dir, capsys):
        """Test that dataset skips slides without features."""
        # Create labels for 4 slides
        csv_path = temp_dir / "labels.csv"
        csv_path.write_text(
            "slide_id,label\n"
            "slide_001,a\n"
            "slide_002,b\n"
            "slide_003,c\n"
            "slide_004,d\n"
        )

        # Only create 2 feature files
        features_dir = temp_dir / "features"
        features_dir.mkdir()
        import h5py
        for slide_id in ['slide_001', 'slide_003']:
            with h5py.File(features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.zeros((10, 64)))

        dataset = MILDataset(csv_path, features_dir)

        # Should only have 2 slides
        assert len(dataset) == 2
        assert 'slide_001' in dataset.slide_ids
        assert 'slide_003' in dataset.slide_ids
        assert 'slide_002' not in dataset.slide_ids


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_case_data_alias(self):
        """Test CaseData alias."""
        from data_loading.dataset import CaseData, GroupedData
        assert CaseData is GroupedData

    def test_case_mil_dataset_alias(self):
        """Test CaseMILDataset alias."""
        from data_loading.dataset import CaseMILDataset, GroupedMILDataset
        assert CaseMILDataset is GroupedMILDataset
