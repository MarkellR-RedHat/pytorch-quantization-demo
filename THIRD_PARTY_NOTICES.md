# Third-party notices

## FlappyLearning

The Quantization Arena's course and bird physics and its game loop structure (`static/js/arena-core.js`, `static/js/arena.js`) are adapted from [FlappyLearning](https://github.com/xviniette/FlappyLearning) by Vincent Bazia. The neuroevolution part of that project is not used; the birds here are flown by policies trained in PyTorch (`arena/train_policy.py`). No images from the original project are included, and all artwork is drawn in code.

```
MIT License

Copyright (c) 2016 Vincent Bazia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Red Hat Display, Red Hat Text, Red Hat Mono

The variable fonts in `static/fonts/` come from [RedHatOfficial/RedHatFont](https://github.com/RedHatOfficial/RedHatFont) and are licensed under the SIL Open Font License 1.1. The full license text is in `static/fonts/OFL.txt`. They are bundled so the demo renders correctly with no network connection.
