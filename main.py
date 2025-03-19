from datasets import load_dataset

# Load the WikiSQL dataset with custom code execution enabled
dataset = load_dataset('wikisql', trust_remote_code=True)

# Print dataset information
print(dataset)

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print the number of examples
print(f"Train examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Test examples: {len(test_data)}")

# Print the first example
print(train_data[0])

from transformers import AutoTokenizer

# Load a tokenizer (e.g., BERT or T5)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the question and SQL query
def preprocess(example):
    question = example['question']
    sql_query = example['sql']['human_readable']
    
    # Tokenize the question
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True)
    
    # Tokenize the SQL query
    labels = tokenizer(sql_query, return_tensors='pt', truncation=True, padding=True)
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels['input_ids']
    }



# Apply preprocessing to the dataset
train_data = train_data.map(preprocess)
val_data = val_data.map(preprocess)
test_data = test_data.map(preprocess)

def encode_table_schema(table):
    header = table['header']
    schema = ', '.join(header)  # Convert column names to a string
    return schema

# Add table schema to the input
def preprocess_with_schema(example):
    question = example['question']
    schema = encode_table_schema(example['table'])
    
    # Combine question and schema
    input_text = f"Translate to SQL: {question} [Schema: {schema}]"
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    
    # Tokenize the SQL query
    sql_query = example['sql']['human_readable']
    labels = tokenizer(sql_query, return_tensors='pt', truncation=True, padding=True)
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels['input_ids']
    }

# Apply preprocessing with schema
train_data = train_data.map(preprocess_with_schema)
val_data = val_data.map(preprocess_with_schema)
test_data = test_data.map(preprocess_with_schema)

from transformers import BertForConditionalGeneration

# Load a pre-trained BERT model for conditional generation
model = BertForConditionalGeneration.from_pretrained('bert-base-uncased')

from torch.utils.data import DataLoader

# Convert to PyTorch tensors
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create DataLoader
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=8)

import torch
from transformers import AdamW

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):  # Number of epochs
    model.train()
    for batch in train_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Validation
    model.eval()
    total_val_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()
    
    print(f"Epoch {epoch + 1}, Val Loss: {total_val_loss / len(val_loader)}")

def evaluate(model, test_loader):
    model.eval()
    exact_match = 0
    total = 0

    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            # Generate SQL query
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Decode ground truth SQL
            ground_truth_sql = tokenizer.decode(labels[0], skip_special_tokens=True)
            
            # Check for exact match
            if generated_sql == ground_truth_sql:
                exact_match += 1
            total += 1

    print(f"Exact Match Accuracy: {exact_match / total * 100:.2f}%")

# Evaluate on the test set
evaluate(model, test_loader)

import sqlite3

def execute_query(sql_query):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create a sample table (replace with your actual table schema)
    cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)')
    cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25)")
    conn.commit()
    
    # Execute the generated SQL query
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error executing query: {e}")
        results = []
    
    conn.close()
    return results

# Example evaluation
generated_sql = "SELECT COUNT(*) FROM users WHERE age > 30"
results = execute_query(generated_sql)
print("Query Results:", results)

# Save the model and tokenizer
model.save_pretrained('text-to-sql-model')
tokenizer.save_pretrained('text-to-sql-model')