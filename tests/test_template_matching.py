import numpy as np
import pytest
import cv2
from src.tools import patch_matching_cross_correlation  # adjust the import as needed

@pytest.fixture(autouse=True)
def seed_random():
    np.random.seed(123)
    
def apply_transformation(image, tx, ty, angle_deg):
    h, w = image.shape
    center = (w / 2, h / 2)
    # Get the rotation matrix around the center
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # Add translation (modify the third column of M)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return transformed, M

def compute_expected_location(point, M):
    x, y = point
    vec = np.array([x, y, 1])
    new_point = M @ vec  # Matrix-vector multiplication
    return new_point  # returns (x_new, y_new)

def compute_ccorel_with_transform(image, template, translation, angle):
    
    tx, ty = translation

    # Apply the transformation to the entire image.
    transformed_image, M = apply_transformation(image, tx, ty, angle)
    
    # Run the patch matching function.
    return transformed_image, M, patch_matching_cross_correlation(
        transformed_image, template, use_convolve=False, pad=False, unnormalized=False
    )
    
def test_patch_matching_translation_and_rotation():
    
    # Generate a random grayscale image (values in 0-255)
    image = (np.random.rand(200, 200) * 255).astype(np.uint8)
    
    # Define a template patch from the original image.
    # For instance, take a 50x50 patch starting at (100, 100) (x, y) coordinates.
    template_top_left = (100, 100)  # (y, x)
    template_size = (50, 50)  # (width, height)
    template = image[template_top_left[1]:template_top_left[1]+template_size[1],
                     template_top_left[0]:template_top_left[0]+template_size[0]]
        
    translation = (10, 15)
    _, M, (corr_map, high_score, pos) = compute_ccorel_with_transform(image, template, translation, 0)

    # Compute where the top-left corner of the template should be in the transformed image.    
    expected_location_xy = compute_expected_location(template_top_left, M)  # (x_new, y_new)
    # Our patch_matching_cross_correlation function returns offsets as (row, col) (i.e. (y, x))
    expected_location = np.array([int(round(expected_location_xy[1])), int(round(expected_location_xy[0]))])

    # a translation only should produce a very high normalized CCOR score    
    np.testing.assert_approx_equal(1.0, high_score, err_msg=f"Cross correlation score should be 1.")
    np.testing.assert_array_equal(pos, expected_location, err_msg=f"Detected offset {pos} does not match expected offset {expected_location}.")

    translation = (10, 15)
    _, M, (corr_map, high_score, pos) = compute_ccorel_with_transform(image, template, translation, 5)

    # Compute where the top-left corner of the template should be in the transformed image.    
    expected_location_xy = compute_expected_location(template_top_left, M)  # (x_new, y_new)
    # Our patch_matching_cross_correlation function returns offsets as (row, col) (i.e. (y, x))
    expected_location = np.array([int(round(expected_location_xy[1])), int(round(expected_location_xy[0]))])

    # a translation with a large rotation should produce a low normalized CCOR score    
    assert 0.0 < high_score < 0.25    
    np.testing.assert_allclose(
        pos, expected_location, atol=2, rtol=2,
        err_msg=f"Detected offset {pos} does not match expected offset {expected_location}."
    )

def test_invalid_match_flipped():
    # Set seed for reproducibility
    np.random.seed(123)
    
    # Generate a random grayscale image (200x200)
    image = (np.random.rand(200, 200) * 255).astype(np.uint8)
    
    # Extract a template patch from a known location in the original image.
    # For example, take a 50x50 patch starting at (80, 80).
    template = image[80:130, 80:130]
    
    # Flip the entire image horizontally to create an invalid match scenario.
    flipped_image = np.fliplr(image)
    
    # Run the patch matching function on the flipped image and the original template.
    corr_map, high_score, pos = patch_matching_cross_correlation(
        flipped_image, template, use_convolve=False, pad=False, unnormalized=False
    )
    
    # For normalized cross-correlation, values range between -1 and 1.
    # Since the template no longer exists in the flipped image, we expect a negative matching score.
    assert high_score < 0.2, f"Expected negative matching score for invalid match, but got {high_score}"

def test_normalized_vs_unnormalized():
    np.random.seed(123)
    
    image = (np.random.rand(100, 100) * 255).astype(np.float32)
    
    # Extract a template patch from the image.
    # For example, use a 30x30 patch starting at (30, 30).
    template_top_left = (30, 30)  # (row, col) order
    template_height, template_width = (30, 30)
    template = image[
        template_top_left[0]:template_top_left[0] + template_height,
        template_top_left[1]:template_top_left[1] + template_width
    ]
    
    # Run patch matching in normalized mode.
    corr_norm, high_score_norm, pos_norm = patch_matching_cross_correlation(
        image, template, use_convolve=False, pad=False, unnormalized=False
    )
    
    # Run patch matching in unnormalized mode.
    corr_unnorm, high_score_unnorm, pos_unnorm = patch_matching_cross_correlation(
        image, template, use_convolve=False, pad=False, unnormalized=True
    )
    
    # For a perfect match in normalized mode, the cross-correlation should be near 1.
    np.testing.assert_allclose(
        high_score_norm, 1.0, atol=1e-3,
        err_msg="Normalized CCOR score should be near 1 for a perfect match."
    )
    
    # Additionally, check that the normalized correlation map's values are within [-1, 1].
    assert corr_norm.min() >= -1.0 and corr_norm.max() <= 1.0, \
        "Normalized correlation map should lie within the range [-1, 1]."
    
    # The unnormalized correlation score should be different (and typically much larger) than 1.
    assert high_score_unnorm > 1.0, "Unnormalized correlation score should not equal 1.0."
    
    # Verify that both versions detect the same location for the best match.
    np.testing.assert_array_equal(
        pos_norm, pos_unnorm,
        err_msg="The detected positions should be identical between normalized and unnormalized matching."
    )

if __name__ == "__main__":
    pytest.main([__file__])
