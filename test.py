from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

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

# Load a tokenizer (e.g., T5)
tokenizer = AutoTokenizer.from_pretrained('t5-small')

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

# Encode table schema
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

# Convert to PyTorch tensors
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Custom collate function
def custom_collate_fn(batch):
    input_ids = [item['input_ids'].squeeze(0) for item in batch]
    attention_mask = [item['attention_mask'].squeeze(0) for item in batch]
    labels = [item['labels'].squeeze(0) for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Create DataLoader with custom collate function
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_data, batch_size=8, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_data, batch_size=8, collate_fn=custom_collate_fn)

# Inspect a batch
batch = next(iter(train_loader))
print("Input IDs shape:", batch['input_ids'].shape)
print("Attention Mask shape:", batch['attention_mask'].shape)
print("Labels shape:", batch['labels'].shape)

# Load a pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Calculate total number of batches
total_batches = len(train_loader)
print(f"Total batches per epoch: {total_batches}")

for epoch in range(3):  # Train for 3 epochs
    print(f"Starting Epoch {epoch + 1}")
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
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
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}/{total_batches}, Loss: {loss.item()}")
    
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
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss}")

# Save the model
model.save_pretrained('text-to-sql-model')
tokenizer.save_pretrained('text-to-sql-model')
