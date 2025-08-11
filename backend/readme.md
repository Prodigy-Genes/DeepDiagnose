
```markdown
# DeepDiagnose Backend

FastAPI backend for medical diagnosis application with PostgreSQL database, JWT authentication, and medical imaging analysis.

## Key Improvements (Latest Updates)

✅ **Fixed BCrypt Installation**  
- Resolved `__about__` attribute error by reinstalling bcrypt with proper compilation  
- Added validation for password hashing reliability

✅ **Enhanced Authentication**  
- Added username uniqueness validation during registration  
- Implemented dual login (email or username) capability  
- Improved error handling for duplicate registrations  

✅ **Multi-Server Architecture**  
- Separated authentication and medical diagnosis into independent services  
- Configured cross-server communication  
- Added port conflict resolution (8000 for auth, 8001 for diagnosis)  

✅ **UUID Serialization**  
- Fixed response validation errors by properly serializing UUID fields  
- Added Pydantic serializer for User model

## System Architecture

```
                          +---------------------+
                          |       Client        |
                          +----------+----------+
                                     |
                    +----------------+----------------+
                    |                                 |
          +---------v---------+             +---------v---------+
          |  Auth Service     |             |  X-Ray Diagnosis  |
          |  Port: 8000       |             |  Port: 8001       |
          +-------------------+             +-------------------+
```

## Features

- 🚀 **Dual FastAPI Servers**: Separate authentication and medical APIs
- 🔐 **JWT authentication** (supports email/username login)
- 🩺 **Medical imaging analysis** with AI models
- 🐘 PostgreSQL database with asyncpg
- 📊 SQLAlchemy ORM with async support

## Installation

1. **Critical BCrypt Installation**:
   ```bash
   pip uninstall -y bcrypt
   pip install --no-binary :all: bcrypt
   ```

2. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Servers

Run in separate terminal windows:

```bash
# Authentication Server (Port 8000)
uvicorn app.auth.main:app --port 8000

# X-Ray Diagnosis Server (Port 8001) 
uvicorn app.diagnosis.main:app --port 8001
```

## API Endpoints

### Authentication Service (`:8000`)
```http
POST /api/auth/register
POST /api/auth/login
```

### Diagnosis Service (`:8001`)
```http
POST /api/diagnosis/predict
GET /api/diagnosis/status
```

## Example Workflow

1. **Get Auth Token**:
```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"doctor1", "password":"secure123"}'
```

2. **Submit X-Ray** (using obtained token):
```bash
curl -X POST http://127.0.0.1:8001/api/diagnosis/predict \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -F "image=@xray.png"
```

## Updated Project Structure

```
backend/
|   # Authentication server
│   ├── app/                     
│   │   ├── api/
│   │   │   └── routes/
│   │   │       └── auth.py
|   |   |   # Diagnosis server
|   |   |   └── xray_diagnosis_api 
│   │   └── core/
│   │       └── security.py
│   └── main.py
│

    
```

## Troubleshooting

**Port Conflicts**:
```bash
# Check running ports
lsof -i :8000
lsof -i :8001

# Kill processes if needed
kill -9 <PID>
```

**Cross-Server Communication**:
- Ensure all requests to diagnosis API include the auth token
- Verify CORS is configured properly if using web clients

**Authentication Issues**:
1. BCrypt errors:
   ```bash
   pip uninstall -y bcrypt passlib
   pip install --no-binary :all: bcrypt passlib
   ```


```