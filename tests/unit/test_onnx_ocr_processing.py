import unittest
import numpy as np

from pcbparse.perception.onnx_ocr_processing import (
    OCRProcessingError, crop_quad, db_polygons, decode_ctc,
    detection_tensor, minimum_rectangle, recognition_tensor, resize_linear,
)


class OCRProcessingTests(unittest.TestCase):
    def test_resize_uses_half_pixel_without_downsample_antialias(self):
        image = np.zeros((2,6,3), dtype=np.uint8)
        image[:,2:4] = 255
        result = resize_linear(image,2,1)
        self.assertTrue(np.array_equal(result,np.zeros((1,2,3),dtype=np.uint8)))
        image = np.array([[[0,0,0],[255,255,255]]],dtype=np.uint8)
        self.assertEqual(resize_linear(image,1,1).tolist(),[[[128,128,128]]])

    def test_detector_and_recognizer_channel_and_padding_contracts(self):
        bgr = np.full((48,100,3),255,dtype=np.uint8)
        bgr[:,:,0] = 0
        detector = detection_tensor(bgr)
        self.assertEqual(detector.shape,(1,3,64,128))
        self.assertAlmostEqual(float(detector[0,0,0,0]),-.485/.229,places=5)
        recognizer = recognition_tensor(bgr)
        self.assertEqual(recognizer.shape,(3,48,320))
        self.assertTrue(np.all(recognizer[0,:,:100] == -1))
        self.assertTrue(np.all(recognizer[1,:,:100] == 1))
        self.assertTrue(np.all(recognizer[:,:,100:] == 0))

    def test_ctc_removes_duplicates_but_keeps_repeats_after_blank(self):
        indices = [1,1,0,1,2,2,0,3]
        probabilities = np.full((1,len(indices),4),.01,dtype=np.float32)
        for i,index in enumerate(indices): probabilities[0,i,index] = .97
        text,score = decode_ctc(probabilities,('', 'A','B',' '))[0]
        self.assertEqual(text,'AAB ')
        self.assertAlmostEqual(score,.97,places=5)
        with self.assertRaises(OCRProcessingError): decode_ctc(probabilities,('', 'A'))
        probabilities[0,0,1] = np.nan
        with self.assertRaises(OCRProcessingError): decode_ctc(probabilities,('', 'A','B',' '))

    def test_rotated_rectangle_area_and_crop_preserve_source_order(self):
        theta = .33
        axes = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        points = np.array([[0,0],[20,0],[20,8],[0,8]])@axes.T+[30,30]
        quad,w,h = minimum_rectangle(points)
        self.assertAlmostEqual(w*h,160,places=5)
        image = np.empty((12,20,3),dtype=np.uint8)
        image[:] = np.arange(20,dtype=np.uint8)[None,:,None]
        cropped = crop_quad(image,np.array([[2,2],[12,2],[12,8],[2,8]],dtype=float))
        self.assertEqual(cropped.shape,(6,10,3))
        self.assertEqual(cropped[0,:,0].tolist(),list(range(2,12)))
        self.assertEqual(quad.shape,(4,2))

    def test_db_separates_regions_and_bounds_expanded_boxes(self):
        probability = np.zeros((40,80),dtype=np.float32)
        probability[3:12,4:25] = .95
        probability[24:34,50:75] = .98
        boxes = db_polygons(probability,width=80,height=40)
        self.assertEqual(len(boxes),2)
        self.assertTrue(all((quad >= 0).all() and (quad[:,0] <= 80).all() and
                            (quad[:,1] <= 40).all() for quad,_ in boxes))
        self.assertLess(boxes[0][0][0,1],boxes[1][0][0,1])
        self.assertGreater(boxes[0][1],.9)
        probability[0,0] = np.inf
        with self.assertRaises(OCRProcessingError): db_polygons(probability,width=80,height=40)

    def test_db_rounds_offset_before_scaling_to_original_image(self):
        probability = np.zeros((40,80),dtype=np.float32)
        probability[10:20,10:30] = .99
        quad,_ = db_polygons(probability,width=113,height=57)[0]
        # Integer contour 10..29 x 10..19 expands by 4.58 raster pixels.
        # The integer round joins give 5..34 x 5..24 before source scaling.
        self.assertEqual(quad.tolist(),[[7.,7.],[48.,7.],[48.,34.],[7.,34.]])

    def test_db_rejects_boxes_collapsed_in_source_pixels_and_crop_stays_strict(self):
        probability = np.zeros((64,64),dtype=np.float32)
        probability[0:4,4:24] = .99
        self.assertEqual(db_polygons(probability,width=20,height=4),[])
        self.assertEqual(db_polygons(probability.T,width=4,height=20),[])
        # Actual medium-model failure: a detector box became a zero-height
        # line on a 4x20 component strip. Direct malformed crop calls fail.
        quad = np.array([[1,0],[12,0],[12,0],[1,0]],dtype=float)
        with self.assertRaisesRegex(OCRProcessingError,'invalid crop size'):
            crop_quad(np.zeros((4,20,3),dtype=np.uint8),quad)

    def test_db_keeps_valid_single_pixel_crops_after_source_rounding(self):
        probability = np.zeros((64,64),dtype=np.float32)
        probability[20:30,10:35] = .99
        for height,width in ((1,40),(40,1),(3,80),(40,80)):
            with self.subTest(height=height,width=width):
                boxes = db_polygons(probability,width=width,height=height)
                self.assertEqual(len(boxes),1)
                crop = crop_quad(np.zeros((height,width,3),dtype=np.uint8),boxes[0][0])
                self.assertGreater(crop.size,0)

    def test_db_accepts_only_one_float32_ulp_of_endpoint_rounding(self):
        probability = np.zeros((40,80),dtype=np.float32)
        probability[10:20,10:30] = 1
        expected = db_polygons(probability,width=80,height=40)
        rounded = probability.copy()
        rounded[10,10] = np.nextafter(np.float32(1), np.float32(np.inf))
        rounded[0,0] = -np.finfo(np.float32).eps
        before = rounded.copy()
        actual = db_polygons(rounded,width=80,height=40)
        self.assertEqual(len(actual),len(expected))
        for (quad,score),(expected_quad,expected_score) in zip(actual,expected):
            np.testing.assert_array_equal(quad,expected_quad)
            self.assertEqual(score,expected_score)
        np.testing.assert_array_equal(rounded,before)
        self.assertTrue(all(0 <= score <= 1 for _,score in actual))

    def test_db_rejects_material_range_errors_and_nonfinite_output(self):
        epsilon = np.finfo(np.float32).eps
        for value in (np.nextafter(np.float32(1)+epsilon,np.float32(np.inf)),
                      np.nextafter(-epsilon,np.float32(-np.inf)),
                      -0.01,1.01,np.nan,np.inf,-np.inf):
            with self.subTest(value=float(value)):
                probability = np.zeros((40,80),dtype=np.float32)
                probability[10:20,10:30] = .95
                probability[0,0] = value
                with self.assertRaises(OCRProcessingError):
                    db_polygons(probability,width=80,height=40)

    def test_budget_failure_is_not_swallowed_inside_bitmap_processing(self):
        def expired(): raise TimeoutError('budget')
        with self.assertRaisesRegex(TimeoutError,'budget'):
            db_polygons(np.zeros((64,64)),width=64,height=64,check=expired)

    def test_large_crop_preserves_rows_and_can_stop_between_blocks(self):
        image = np.repeat(np.arange(128,dtype=np.uint8)[:,None,None],100,axis=1)
        image = np.repeat(image,3,axis=2)
        quad = np.array([[0,0],[100,0],[100,128],[0,128]],dtype=float)
        self.assertTrue(np.array_equal(crop_quad(image,quad),image))
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 2: raise TimeoutError('crop budget')
        with self.assertRaisesRegex(TimeoutError,'crop budget'):
            crop_quad(image,quad,check=check)
        self.assertEqual(len(calls),2)


if __name__ == '__main__': unittest.main()
