from flask import request, jsonify
from app.bpe import BPEProcessor

def init_routes(app):
    @app.route('/bpe', methods=['POST'])
    def bpe():
        data = request.get_json()
        print(data)
        text = data.get('text')
        num_merges = data.get('num_merges', 10)  
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        bpe_processor = BPEProcessor()
        encoded_tokens, decoded_text, vocab_size, num_merges_done = bpe_processor.process(text, num_merges)
        
        return jsonify({
            'encoded_tokens': encoded_tokens,
            'decoded_text': decoded_text,
            'vocab_size': vocab_size,
            'num_merges': num_merges_done
        })
