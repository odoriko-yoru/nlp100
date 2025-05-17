#!/bin/python
import glob
import os
import re
import shutil
import sys

twitter_card = """
    <meta name="twitter:card" content="summary">
    <meta name="twitter:site" content="@chokkanorg">
    <meta name="twitter:title" content="言語処理100本ノック">
    <meta name="twitter:description" content="言語処理100本ノックは、実用的でワクワクするような課題に取り組みながら、自然言語処理、大規模言語モデル、プログラミング、研究のスキルを楽しく習得することを目指した問題集です。">
    <meta name="twitter:image" content="https://nlp100.github.io/2025/_images/nlp100-2025-ja.jpg">
    </head>
"""

sagemaker_studio_lab = """
<span class="btn__text-container">Colab</span>
</a>
</li>

<li>
    <a href="https://studiolab.sagemaker.aws/import/github/nlp100/2025/blob/main/ja/{}"
       class="btn btn-sm dropdown-item"
       title="Launch on SageMaker Studio Lab"
       data-bs-placement="left" data-bs-toggle="tooltip">

<span class="btn__icon-container">
  <i class="fab fa-aws"></i>
</span>
<span class="btn__text-container">SageMaker</span>
"""

def modify_html():
    # Modify the generated HTML files.
    for src in glob.glob('_build/html/**/*.html', recursive=True):
        print(f'Updating: {src}')
        
        # Load the HTML content.
        with open(src) as fi:
            content = fi.read()
            
        # Find the path to .ipynb
        path = ''
        m = re.search(r'"https://colab.research.google.com/github/nlp100/2025/blob/main/ja/([^"]+)"', content)
        if m is not None:
            path = m.group(1)
            print(f'    path: {path}')
        
        # Add meta tags for Twitter card.
        content = content.replace('</head>', twitter_card)
        
        # Add the button for SageMaker Studio Lab.
        if path:
            content = content.replace(
                '<span class="btn__text-container">Colab</span>',
                sagemaker_studio_lab.format(path)
                )
        
        # Write out the HTML content.
        with open(src, 'w') as fo:
            fo.write(content)

if __name__ == '__main__':
    modify_html()
