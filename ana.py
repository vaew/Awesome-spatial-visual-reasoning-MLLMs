import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def extract_titles(content):
    titles = []
    # 匹配形如 "1. [arxiv 2406, ICRA'25] 标题" 或 "* [arxiv 2406, ICRA'25] 标题" 的行
    pattern = r'^\d+\.\s*\[.*?\]\s*(.*?)(?=\s*\[|$)|^\*\s*\[.*?\]\s*(.*?)(?=\s*\[|$)'
    
    for line in content.split('\n'):
        match = re.search(pattern, line)
        if match:
            # 获取匹配的标题（可能是第一个或第二个捕获组）
            title = match.group(1) or match.group(2)
            if title:
                titles.append(title.strip())
    
    return titles

def clean_title(title):
    # 移除不需要的词
    stop_words = [
        'Code', 'Dataset', 'arxiv', 'Arxiv', 'Code will be released soon', 
        'Datasets', 'Code will be released', 'will be released soon',
        'CVPR', 'ICRA', 'NAACL', 'EMNLP', 'CoRL', 'NIPS', 'ICLR', 'ICML',
        'CVPR\'24', 'ICRA\'25', 'NAACL\'25', 'EMNLP\'24', 'CoRL\'24', 'NIPS\'23',
        'ICLR\'24', 'ICML\'24', 'CVPR\'25', 'ICRA\'24', 'NAACL\'24', 'EMNLP\'25',
        'CoRL\'25', 'NIPS\'24', 'ICLR\'25', 'ICML\'25'
    ]
    
    for word in stop_words:
        title = title.replace(word, '')
    
    # 移除方括号和链接
    title = re.sub(r'\[.*?\]', '', title)
    # 移除多余的空格
    title = ' '.join(title.split())
    return title

def main():
    # 读取markdown文件
    with open('Awesome-spatial-visual-reasoning.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有标题
    titles = extract_titles(content)
    
    # 打印原始标题
    print("\n原始标题:")
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")
    
    # 清理标题
    cleaned_titles = [clean_title(title) for title in titles]
    
    # 打印清理后的标题
    print("\n清理后的标题:")
    for i, title in enumerate(cleaned_titles, 1):
        print(f"{i}. {title}")
    
    # 生成词云
    text = ' '.join(cleaned_titles)
    
    if not text.strip():
        print("Error: No text available for word cloud generation")
        return
    
    # 生成词云
    wordcloud = WordCloud(
        width=1200, 
        height=800,
        background_color='white',
        max_words=150,
        collocations=False
    )
    
    wordcloud.generate(text)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    # plt.title('Research Topics in Spatial-Visual Reasoning', fontsize=20, pad=20)
    plt.tight_layout(pad=0)
    plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()