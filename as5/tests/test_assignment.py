
import imageio.v2 as imageio
import numpy as np
from tests.utils import replace_stdin
from as5.as5 import run, load_images, convert_gray_luminance, binarize_images

class TestAssignment:
    def test_all_cases(self, capsys):
        inp = 'tests/resources/testcases/2.in'
        exp_out = ''
        with open('tests/resources/testcases/2.out', 'r') as f:
            exp_out = f.readlines()
        exp_out = ''.join(exp_out)
        
        with replace_stdin(open(inp, 'r')):
            run()
            captured = capsys.readouterr()
            out = captured.out
        print(out)
        assert out == exp_out
    
    def test_load_images(self):
        image_names = ['pedras1.png']
        exp_img = imageio.imread('tests/resources/InputImages/pedras1.png')
        img = load_images(image_names)[0]
        assert np.array_equal(exp_img, img)
    
    def test_convert_gray(self):
        orig_img = imageio.imread('tests/resources/InputImages/pedras1.png')
        convert_gray_luminance([orig_img])
    
    def test_binarize(self):
        T = 64
        orig_img = imageio.imread('tests/resources/InputImages/pedras1.png')
        binarize_images([orig_img], T)