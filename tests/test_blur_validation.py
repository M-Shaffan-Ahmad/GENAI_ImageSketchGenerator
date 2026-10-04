import unittest
import numpy as np
from task1.blur_validation import fixed_manifest, summarize
from corruptions import apply_corruption


class BlurValidationTests(unittest.TestCase):
    def test_same_photos_and_exact_fixed_severities(self):
        sources = [dict(source_id=i,path=f'image_{i}.jpg',sha256='example') for i in [10,20]]
        rows = fixed_manifest(sources)
        self.assertEqual(rows,fixed_manifest(sources))
        for i in [10,20]:
            cases = [r for r in rows if r['source_id']==i]
            self.assertEqual([r['severity'] for r in cases],['none','low','medium','high'])
            self.assertEqual([(r['parameters']['ksize'],r['parameters']['sigma']) for r in cases[1:]],[(3,.7),(5,1.5),(7,2.5)])
        image = np.random.default_rng(42).random((128,128,3)).astype(np.float32)
        for row in rows:
            np.testing.assert_array_equal(apply_corruption(image,row),apply_corruption(image,row))

    def test_paired_improvement_counts(self):
        rows = []
        for severity in ['none','low','medium','high']:
            for dp,ds in [(1,.1),(-3,-.2),(2,-.1)]:
                rows.append(dict(severity=severity,input_psnr=20,psnr=20+dp,psnr_delta=dp,
                                 input_ssim=.8,ssim=.8+ds,ssim_delta=ds,input_l1=.1,l1=.2,l1_improvement=-.1))
        result = summarize(rows)
        self.assertEqual(result['low']['psnr_improved_n'],2)
        self.assertEqual(result['low']['both_improved_n'],1)
        self.assertAlmostEqual(result['low']['psnr_delta'],0)


if __name__ == '__main__':
    unittest.main()
