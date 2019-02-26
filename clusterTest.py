import DocumentAnalyzer
import sys


def main():
    path = 'D:\\DATA\\smaller_splited_dataset2\\test\\8140.jpg'
    analyzer = DocumentAnalyzer.DocumentAnalyzer(0.23)
    analyzer.get_document_paragraphs(path)

    return 0

if __name__ == "__main__":
    sys.exit(main())
