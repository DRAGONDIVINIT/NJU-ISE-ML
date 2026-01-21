import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import matplotlib.pyplot as plt

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 加载数据集
datasets = load_dataset('glue', 'sst2')
train_data = datasets['train']
valid_data = datasets['validation']

# 提取文本和标签
train_texts = [example['sentence'] for example in train_data]
train_labels = [example['label'] for example in train_data]
valid_texts = [example['sentence'] for example in valid_data]
valid_labels = [example['label'] for example in valid_data]

# 划分标注数据和未标注数据（10%标注数据）
labeled_size = int(0.1 * len(train_texts))
indices = np.arange(len(train_texts))
np.random.shuffle(indices)

labeled_indices = indices[:labeled_size]
unlabeled_indices = indices[labeled_size:]

labeled_texts = [train_texts[i] for i in labeled_indices]
labeled_labels = [train_labels[i] for i in labeled_indices]
unlabeled_texts = [train_texts[i] for i in unlabeled_indices]

# 获取未标注数据的真实标签（用于分析）
unlabeled_true_labels = [train_labels[i] for i in unlabeled_indices]

# 向量化文本数据
vectorizer = CountVectorizer()
X_labeled = vectorizer.fit_transform(labeled_texts)
X_unlabeled = vectorizer.transform(unlabeled_texts)
X_valid = vectorizer.transform(valid_texts)

def semi_supervised_em_iterations(n_iterations, confidence_threshold=0.7):
    """运行半监督EM算法，返回每轮的性能指标"""
    
    print(f"\n{'='*60}")
    print(f"迭代次数: {n_iterations}")
    print('='*60)
    
    # 初始化模型
    model = MultinomialNB()
    model.fit(X_labeled, labeled_labels)
    
    # 记录性能指标
    results = {
        'iterations': [],
        'valid_accuracies': [],
        'pseudo_label_error_rates': [],
        'selected_samples': [],
        'cumulative_samples': []
    }
    
    # 当前已选择的样本索引
    selected_indices = set()
    
    for iteration in range(n_iterations):
        print(f"\n第 {iteration+1}/{n_iterations} 轮迭代:")
        
        # E步：预测未标注数据
        proba = model.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        
        # 筛选高置信度样本（排除已选择的）
        available_indices = np.array([i for i in range(len(unlabeled_texts)) if i not in selected_indices])
        if len(available_indices) == 0:
            print("所有未标注数据都已使用，停止迭代")
            break
            
        available_proba = max_proba[available_indices]
        
        # 筛选置信度高于阈值的样本
        high_conf_mask = available_proba >= confidence_threshold
        new_indices = available_indices[high_conf_mask]
        
        if len(new_indices) == 0:
            print("没有新的高置信度样本，停止迭代")
            break
        
        # 更新已选择样本
        selected_indices.update(new_indices)
        
        # 获取伪标签
        pseudo_labels = model.predict(X_unlabeled[list(new_indices)])
        
        # 计算伪标签错误率（仅本轮新样本）
        true_labels_new = [unlabeled_true_labels[i] for i in new_indices]
        error_rate_new = 1 - accuracy_score(true_labels_new, pseudo_labels)
        
        # 计算所有已选择样本的错误率
        if selected_indices:
            all_pseudo = model.predict(X_unlabeled[list(selected_indices)])
            all_true = [unlabeled_true_labels[i] for i in selected_indices]
            error_rate_all = 1 - accuracy_score(all_true, all_pseudo)
        else:
            error_rate_all = 0.0
        
        # 验证集性能
        valid_pred = model.predict(X_valid)
        valid_acc = accuracy_score(valid_labels, valid_pred)
        
        print(f"  本轮新增样本数: {len(new_indices)}")
        print(f"  累计选择样本数: {len(selected_indices)}")
        print(f"  本轮伪标签错误率: {error_rate_new:.4f}")
        print(f"  累计伪标签错误率: {error_rate_all:.4f}")
        print(f"  验证集准确率: {valid_acc:.4f}")
        
        # 保存结果
        results['iterations'].append(iteration+1)
        results['valid_accuracies'].append(valid_acc)
        results['pseudo_label_error_rates'].append(error_rate_new)
        results['selected_samples'].append(len(new_indices))
        results['cumulative_samples'].append(len(selected_indices))
        
        # M步：重新训练模型
        if selected_indices:
            X_combined = np.vstack([X_labeled.toarray(), X_unlabeled[list(selected_indices)].toarray()])
            all_pseudo_labels = np.concatenate([labeled_labels, 
                                               model.predict(X_unlabeled[list(selected_indices)])])
            model = MultinomialNB()
            model.fit(X_combined, all_pseudo_labels)
    
    return results

# 测试不同迭代次数
iterations_to_test = [3, 5, 15]
all_results = {}

for n_iter in iterations_to_test:
    results = semi_supervised_em_iterations(n_iter, confidence_threshold=0.7)
    all_results[n_iter] = results


# 最终性能对比
print("\n" + "="*70)
print("不同迭代次数的最终性能对比")
print("="*70)

for n_iter in iterations_to_test:
    if all_results[n_iter]['valid_accuracies']:
        final_acc = all_results[n_iter]['valid_accuracies'][-1]
        best_acc = max(all_results[n_iter]['valid_accuracies'])
        best_iter = all_results[n_iter]['iterations'][all_results[n_iter]['valid_accuracies'].index(best_acc)]
        
        avg_error_rate = np.mean(all_results[n_iter]['pseudo_label_error_rates'])
        total_samples = all_results[n_iter]['cumulative_samples'][-1] if all_results[n_iter]['cumulative_samples'] else 0
        
        print(f"\n迭代次数: {n_iter}")
        print(f"  最终验证准确率: {final_acc:.4f}")
        print(f"  最佳验证准确率: {best_acc:.4f} (第{best_iter}轮)")
        print(f"  平均伪标签错误率: {avg_error_rate:.4f}")
        print(f"  累计使用伪标签样本数: {total_samples}")

# 分析迭代次数对性能的影响
print("\n" + "="*70)
print("迭代次数对模型性能的影响分析")
print("="*70)

print("""
1. 性能变化趋势分析：

   a) 早期迭代（1-5轮）：
      - 模型性能通常快速提升
      - 伪标签错误率逐渐下降（模型学习能力增强）
      - 每轮新增样本数较多（高质量易分类样本被优先选择）
      
   b) 中期迭代（5-10轮）：
      - 性能提升速度放缓
      - 伪标签错误率可能稳定或轻微上升
      - 新增样本数减少（剩余样本更难分类）
      
   c) 后期迭代（10轮以上）：
      - 性能可能达到饱和或开始下降
      - 伪标签错误率可能显著上升
      - 新增样本质量下降

2. 过多迭代导致性能下降的原因：

   a) 错误累积与误差传播：
      - 早期引入的错误伪标签会在后续迭代中被强化
      - 错误标签污染训练数据，导致模型学习错误的模式
      - 错误样本可能产生更多错误预测，形成恶性循环
      
   b) 低质量样本引入：
      - 随着迭代进行，剩余的未标注样本往往是：
        1) 分类边界附近的模糊样本
        2) 噪声较大或特征不明显的样本
        3) 模型难以正确分类的困难样本
      - 对这些样本赋予伪标签的错误率较高
      
   c) 过拟合伪标签：
      - 模型过度拟合到伪标签的噪声
      - 特别是在伪标签样本远多于真实标注样本时
      - 模型可能学到伪标签中的系统性偏差
      
   d) 置信度阈值失效：
      - 后期迭代中，即使是高置信度的预测也可能是错误的
      - 模型可能对某些错误模式过于"自信"
      - 置信度校准失效
      
   e) 数据分布偏移：
      - 伪标签样本的分布可能与真实分布存在偏差
      - 随着迭代进行，这种偏差可能被放大
      - 导致模型在真实测试数据上泛化能力下降

3. 最佳迭代次数的确定因素：

   a) 数据特征：
      - 数据质量高、噪声少时，可进行更多迭代
      - 类别分布均衡时更稳定
      
   b) 模型容量：
      - 复杂模型可能更快过拟合伪标签噪声
      - 简单模型对噪声更鲁棒，可支持更多迭代
      
   c) 初始标注数据量：
      - 初始标注数据越多，基础模型越可靠
      - 可支持更多轮次迭代而不严重退化
      
   d) 停止准则建议：
      1) 验证集性能连续下降
      2) 伪标签错误率显著上升（>0.3）
      3) 新增样本质量过低（置信度下降）
      4) 模型预测过于自信（概率分布熵过低）

4. 改进策略建议：

   a) 早停机制：
      - 监控验证集性能，在性能下降前停止
      - 使用滑动窗口平均检测下降趋势
      
   b) 动态阈值调整：
      - 随迭代降低阈值要求，但增加质量验证
      - 对后期样本进行更严格的筛选
      
   c) 伪标签验证：
      - 定期重新评估已添加的伪标签
      - 移除置信度下降或预测不一致的样本
      
   d) 模型集成：
      - 使用多个模型投票决定伪标签
      - 减少单个模型的错误影响
      
   e) 课程学习：
      - 从易到难逐步增加样本
      - 后期专注于高质量样本，而非数量
""")