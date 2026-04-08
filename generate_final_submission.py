import pandas as pd

TEMPLATE_PATH = '数据与样例/submit_example.csv'
SOURCE_PATH = 'submission_task1.csv'
OUTPUT_PATH = 'submission_task1_final.csv'


def main():
    template = pd.read_csv(TEMPLATE_PATH)
    source = pd.read_csv(SOURCE_PATH)

    if list(template.columns) != ['TIME', 'V']:
        raise ValueError('submit_example.csv 列名应为 TIME,V')
    if list(source.columns) != ['TIME', 'V']:
        raise ValueError('submission_task1.csv 列名应为 TIME,V')

    n = len(template)
    if len(source) < n:
        raise ValueError(f'源数据长度不足：{len(source)} < {n}')

    # 用已有预测结果最后 n 条作为 61 天（11-12月）预测值
    values = source['V'].tail(n).to_numpy()

    final_submission = pd.DataFrame({
        'TIME': template['TIME'],
        'V': values
    })

    final_submission.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f'模板行数: {len(template)}')
    print(f'源预测行数: {len(source)}')
    print(f'输出文件: {OUTPUT_PATH}')
    print('输出时间范围:', final_submission['TIME'].iloc[0], '->', final_submission['TIME'].iloc[-1])
    print(final_submission.head(3).to_string(index=False))
    print(final_submission.tail(3).to_string(index=False))


if __name__ == '__main__':
    main()
