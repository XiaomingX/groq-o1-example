本文将带你一步步实现一个基于 **Groq API** 的推理助手，该助手能够高效地模拟接近chatgpt-o1的推理逻辑。通过该教程，你将掌握以下关键知识点：

## 一、Groq API 基础

### 1.1 安装依赖

确保你的环境中安装了以下库：

```bash
pip install groq-python dotenv
```

还需要确保你已经获取了 Groq API 的密钥，并将其保存在 `.env` 文件中：

```env
GROQ_API_KEY=your_api_key_here
```

### 1.2 初始化客户端

通过 `dotenv` 加载环境变量，创建一个 Groq API 客户端：

```python
import groq
from dotenv import load_dotenv
load_dotenv()

class GroqAPIClient:
    def __init__(self, custom_client=None):
        self.client = custom_client or groq.Groq()
```

***

## 二、实现等同于chatgpt-o1的推理模型

### 2.1 构建推理助手类

**核心逻辑**：推理助手通过 API 循环调用模拟逐步推理的过程，每一步都包含以下内容：

*   当前步骤的标题（`title`）
*   推理内容（`content`）
*   是否需要进入下一步或返回最终答案（`next_action`）

#### 推理助手核心代码

```python
class ReasoningAssistant:
    def __init__(self, api_client):
        self.api_client = api_client

    def generate_response(self, prompt):
        messages = self._initialize_messages(prompt)
        steps = []
        step_count = 1
        total_thinking_time = 0

        while True:
            start_time = time.time()
            step_data = self.api_client.make_api_call(messages, 300)
            end_time = time.time()
            thinking_time = end_time - start_time
            total_thinking_time += thinking_time

            steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
            messages.append({"role": "assistant", "content": json.dumps(step_data)})

            if step_data['next_action'] == 'final_answer' or step_count > 25:  # 防止死循环
                break

            step_count += 1
            yield steps, None

        final_answer = self._generate_final_answer(messages)
        steps.append(("Final Answer", final_answer['content'], final_answer['thinking_time']))
        yield steps, total_thinking_time + final_answer['thinking_time']

        self._save_to_markdown(prompt, steps)
```

***

### 2.2 API 调用及容错机制

通过 `make_api_call` 方法封装 API 调用逻辑，并添加了重试机制：

```python
class GroqAPIClient:
    def make_api_call(self, messages, max_tokens, is_final_answer=False):
        for attempt in range(3):  # 最多尝试三次
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"} if not is_final_answer else None
                )
                if is_final_answer:
                    return response.choices[0].message.content
                else:
                    return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    error_content = {
                        "title": "Error",
                        "content": f"Failed after 3 attempts. Error: {str(e)}"
                    }
                    return error_content
                time.sleep(1)  # 重试前等待 1 秒
```

***

### 2.3 设计消息格式

推理过程中消息的格式如下：

1.  **系统消息**：定义模型行为和推理规则。
2.  **用户消息**：传递用户的输入。
3.  **助手消息**：记录每步推理的结果。

初始化消息的方法如下：

```python
def _initialize_messages(self, prompt):
    return [
        {"role": "system", "content": self._system_message()},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
```

系统消息的定义：

```python
def _system_message(self):
    return (
        """
        You are an expert AI assistant that explains your reasoning step by step...
        """
    )
```

***

### 2.4 保存推理过程

将推理过程以 Markdown 文件的形式保存：

```python
def _save_to_markdown(self, prompt, steps):
    filename = f"{prompt[:20]}.md".replace(" ", "_").replace("/", "-")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# Problem: {prompt}\n\n")
        for step in steps:
            file.write(f"## {step[0]}\n\n{step[1]}\n\n(Thinking time: {step[2]:.2f} seconds)\n\n")
```

***

## 三、运行推理模型

编写 `main` 函数，启动推理过程：

```python
def main():
    custom_client = None  # 替换为实际的 Groq 客户端
    api_client = GroqAPIClient(custom_client)
    assistant = ReasoningAssistant(api_client)

    prompt = "I am playing Werewolf and I am the Hunter. It is the second round..."
    for steps, total_time in assistant.generate_response(prompt):
        for step in steps:
            print(f"{step[0]}:\n{step[1]}\n(Thinking time: {step[2]:.2f} seconds)\n")
        if total_time is not None:
            print(f"Total Thinking Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
```

***

## 四、教程总结

1.  **模块化设计**：将 API 调用、推理逻辑和结果存储分离，增强代码的可维护性。
2.  **容错机制**：通过重试机制提高调用的可靠性。
3.  **结果可视化**：将推理过程保存为 Markdown 文件，方便分析和调试。

按照此教程构建的推理模型不仅高效，还具有强大的可扩展性，可应用于复杂推理场景。

# 使用示例-1（做高考和大学的数学题、化学题、物理题）

# Problem: Given the parabola ( y^2 = 16x ), the coordinates of the focus are \_\_\_\_\_\_\_\_.

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/711b62fa962a4443871d1e1635a8d54b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgdWhha2Fkb3Rjb20=:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjg3NTk3ODE0NzY5MjkxMCJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732434264&x-orig-sign=UfcqiR48U%2Fs3eWKFHTjOhc9ReIM%3D)

## Step 1: Decompose the Problem

The given equation is ( y^2 = 16x ), which is a parabola in the standard form ( y^2 = 4px ). The goal is to find the coordinates of the focus. To do this, we need to identify the value of ( p ) and use it to determine the coordinates of the focus.

(Thinking time: 1.35 seconds)

## Step 2: Identify the Value of p

Comparing the given equation ( y^2 = 16x ) to the standard form ( y^2 = 4px ), we can see that ( 4p = 16 ). Solving for ( p ), we get ( p = 16/4 = 4 ). Therefore, the value of ( p ) is 4.

(Thinking time: 0.73 seconds)

## Step 3: Determine the Coordinates of the Focus

The coordinates of the focus of a parabola in the standard form ( y^2 = 4px ) are ( p, 0 ). Since we found that ( p = 4 ), the coordinates of the focus are ( 4, 0 ). However, we should also consider alternative approaches to verify our answer.

(Thinking time: 0.79 seconds)

## Step 4: Alternative Approach Using the Definition of a Parabola

A parabola is defined as the set of all points that are equidistant to the focus and the directrix. The directrix of a parabola in the standard form ( y^2 = 4px ) is the line ( x = -p ). Since we found that ( p = 4 ), the directrix is the line ( x = -4 ). Using this information, we can verify that the coordinates of the focus are indeed ( 4, 0 ).

(Thinking time: 1.66 seconds)

## Step 5: Alternative Approach Using the Equation of the Parabola

We can also use the equation of the parabola to find the coordinates of the focus. The equation ( y^2 = 16x ) can be rewritten as ( x = y^2/16 ). This equation represents a parabola that opens to the right, and the focus is located at a distance ( p ) from the vertex. Since we found that ( p = 4 ), the coordinates of the focus are indeed ( 4, 0 ).

(Thinking time: 1.61 seconds)

## Step 6: Re-examine the Answer Using a Different Method

Let's re-examine our answer by using the formula for the focal length of a parabola, which is given by ( c = 1/(4a) ) for a parabola in the form ( y^2 = 4ax ). Comparing this to our equation ( y^2 = 16x ), we can see that ( 4a = 16 ), so ( a = 4 ). Plugging this into the formula, we get ( c = 1/(4\*4) = 1/16 ), but this is not what we are looking for. However, we can see that ( c = p = 4 ) is actually the focal length, not the focal coordinate. The focal coordinate is ( p, 0 ) = ( 4, 0 ).

(Thinking time: 1.33 seconds)

## Final Answer

The coordinates of the focus are (4, 0).

(Thinking time: 0.61 seconds)

# 使用示例-2（ 玩狼人杀或者天黑请闭眼，进行逻辑推理 ）

# 问题描述：我正在玩狼人杀，我是猎人。现在是第二轮，预言家已经死了。在我发言之前，有人宣称自己是猎人，但实际上不是。轮到我发言时，我应该如何应对？

## 第一步：理解问题

当前的情况是，我是猎人，但在我之前有人虚假宣称自己是猎人。而且预言家已经阵亡，当前是第二轮。我需要找到最佳的应对策略，决定在自己发言时该怎么做。

## 第二步：分析局势

由于预言家已死，我作为猎人无法得知谁是狼人。虚假的猎人宣称者可能是狼人，企图误导村民；也可能是普通村民，为了保护自己而撒谎。我需要判断对方的动机，考虑怎样应对才能最小化暴露自己的身份。

## 第三步：可能的应对方式

猎人有几种选择：1）揭穿对方并表明自己才是真正的猎人，2）顺着对方的话，假装相信，或者3）保持沉默，观察其他玩家的反应。每种选择都有利弊：揭穿对方可能会让狼人知道我的身份，而顺着对方的话则可以获得更多关于其动机的信息。

## 第四步：评估暴露身份的风险

如果我暴露了猎人的身份，下一轮可能会成为狼人的攻击目标。但如果保持沉默，可能会错失从虚假猎人身上获得有用信息的机会。我需要权衡暴露身份的风险与可能获得的信息之间的利弊。

## 第五步：考虑其他视角

假设虚假的猎人是一个为了自保的普通村民，或者是一个试图制造混乱的狼人，甚至可能是另一个猎人的同伴（如果游戏中有多个猎人）。我需要从不同的角度考虑，才能更好地判断对方的真实意图。

## 第六步：换个思路重新审视问题

与其专注于揭穿虚假的猎人，不如考虑猎人的主要目标是找出狼人。我可以利用当前的机会，通过提问来获取关于其他玩家的信息，而不直接回应虚假的猎人宣称。这样可以帮助我更好地了解当前的游戏局势，为后续行动做好准备。

## 第七步：综合信息得出结论

经过对局势、应对方式、风险以及不同视角的分析，我认为猎人应该采取一种既能获取信息又能降低风险的策略。具体来说，可以在发言时提出问题，试图了解其他玩家的行为，而不直接揭穿虚假的猎人宣称。这样既能隐藏自己的身份，又能收集更多关于游戏状态的信息。

## 最终结论

轮到你发言时，建议你通过提问来获取其他玩家的信息，而不直接回应虚假的猎人宣称。例如，你可以让大家讨论前一晚的情况，或者询问某些玩家的行为动机。通过这样做，你可以观察其他人的反应，收集有用的信息，为后续决策提供依据，同时尽量隐藏你猎人的真实身份。
