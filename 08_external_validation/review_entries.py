#!/usr/bin/env python3
import csv
import sys

def load_entries(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    reader = csv.reader(content.splitlines())
    rows = list(reader)
    header = rows[0]
    entries = rows[1:]
    return header, entries

def display_entry(idx, entry, header):
    print(f"\n{'='*60}")
    print(f"条目 {idx+1}: report_id = {entry[0]}")
    print(f"{'='*60}")
    print(f"就诊号: {entry[1]}")
    print(f"检查项目: {entry[2]}")
    print(f"检查时间: {entry[3]}")
    print(f"rule_label: {entry[4]}")
    print(f"deepseek_verdict: {entry[5]}")
    print(f"deepseek_reason: {entry[6]}")
    print(f"glm_verdict: {entry[7]}")
    print(f"glm_reason: {entry[8]}")
    print(f"y_status_flips: {entry[9]}")
    print(f"\n报告文本:")
    print(f"{entry[10]}")
    print(f"\n当前human_verdict: {entry[11]}")
    print(f"{'='*60}")

def main():
    filename = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval/llm_discordant_for_review.csv'
    header, entries = load_entries(filename)
    
    print(f"共加载 {len(entries)} 条记录")
    print("输入 'y' 判定为阳性, 'n' 判定为阴性, 's' 跳过, 'q' 退出")
    print("输入数字可跳转到指定条目")
    
    idx = 0
    while idx < len(entries):
        display_entry(idx, entries[idx], header)
        
        try:
            user_input = input(f"\n您的判断 (条目 {idx+1}/{len(entries)}): ").strip().lower()
            
            if user_input == 'q':
                print("退出审核")
                break
            elif user_input == 'y':
                print(f"已记录: 条目 {idx+1} 判定为 阳性")
                idx += 1
            elif user_input == 'n':
                print(f"已记录: 条目 {idx+1} 判定为 阴性")
                idx += 1
            elif user_input == 's':
                print(f"已跳过: 条目 {idx+1}")
                idx += 1
            elif user_input.isdigit():
                target = int(user_input) - 1
                if 0 <= target < len(entries):
                    idx = target
                else:
                    print(f"请输入 1-{len(entries)} 之间的数字")
            else:
                print("无效输入，请输入 y/n/s/q 或数字")
        except KeyboardInterrupt:
            print("\n\n退出审核")
            break
        except EOFError:
            print("\n\n退出审核")
            break
    
    print("\n审核结束")

if __name__ == "__main__":
    main()
