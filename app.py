from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)


def load_search_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


DATA_FILE = 'graph_with_updated_rewards_and_weights.json'
search_data = load_search_data(DATA_FILE)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.args.get('query', '').lower().strip()
    if not query:
        return jsonify([])

        # 前缀匹配搜索结果
    results = [item for item in search_data if item['name'].lower().startswith(query)]
    return jsonify(results[:10])  # 返回前10个结果


def wayPoint_add():
    wayPoint = []
    if request.method == 'POST':
        wayPoint.append("")
        name = request.form.get('name')
        greeting = f'你好，{name}!'
    return render_template('index.html', greeting=greeting)


if __name__ == '__main__':
    app.run(debug=True)
