from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import uuid
import pandas as pd

# import the ml model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

app = FastAPI()

#  cross-origin requests for development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# jinja for loop used
templates = Jinja2Templates(directory="fronted")
# Serve frontend static files under /static to avoid shadowing API routes
app.mount("/static", StaticFiles(directory="fronted"), name="static")


@app.get('/', include_in_schema=False)
async def root_index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            # request from tier data
            "request": request,
            "tier_1_cities": tier_1_cities,
            "tier_2_cities": tier_2_cities
        }
    )


@app.get('/predict.html', include_in_schema=False)
async def predict_page():
    return FileResponse('fronted/Predict.html')

# pydantic model to validate incoming data
class UserInput(BaseModel):

    age: Annotated[int,Field(...,gt=0,lt=120,description="Age must be between 1 and 119")]
    weight: Annotated[float,Field(...,gt=0,description="Weight must be a positive number")]
    height:Annotated[float,Field(...,gt=0,lt=2.5,description="Height must be a positive number")]
    income_lpa:Annotated[float,Field(...,gt=0,description="Annual salary in income of lpa ")]   
    smoker: Annotated[bool,Field(...,description="Smoker must be 'yes' or 'no'")]
    city: Annotated[str,Field(...,description="The city that the user by belogs to")]
    occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
                'business_owner', 'unemployed', 'private_job'],Field(...,description="The occupation of the user")]


    
    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight/(self.height**2)
    
    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"
        
    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"
    
    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
        


@app.post('/predict')
async def predict_premium(request: Request):
    
    content_type = (request.headers.get('content-type') or '').lower()

    if 'application/json' in content_type:
        # JSON API client
        try:
            body = await request.json()
            data = UserInput(**body)
        except Exception as e:
            return JSONResponse(status_code=400, content={'detail': f'Invalid JSON body: {e}'})
        is_json = True
    else:
        # HTML form submission
        try:
            form = await request.form()
            fv = {
                'age': int(form.get('age')),
                'weight': float(form.get('weight')),
                'height': float(form.get('height')),
                'income_lpa': float(form.get('income_lpa')),
                'smoker': str(form.get('smoker')).lower() in ('true', 'on', '1', 'yes'),
                'city': form.get('city'),
                'occupation': form.get('occupation')
            }
            data = UserInput(**fv)
        except Exception as e:
            return JSONResponse(status_code=400, content={'detail': f'Invalid form data: {e}'})
        is_json = False

    # compute features and prediction
    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    prediction = model.predict(input_df)[0]

    # include the computed features back in the response so the frontend can display details
    features = input_df.to_dict(orient='records')[0]
    for k, v in list(features.items()):
        if isinstance(v, float):
            features[k] = round(v, 2)

    original_input = data.dict()
    response_content = {
        'predicted_category': prediction,
        'features': features,
        'input': original_input
    }

    if not is_json:
        rid = str(uuid.uuid4())
        STORE[rid] = response_content
        return RedirectResponse(url=f'/predict.html?rid={rid}', status_code=303)

    return JSONResponse(status_code=200, content=response_content)


# simple mermory storage
STORE = {}


@app.get('/result/{rid}')
def get_result(rid: str):
    val = STORE.get(rid)
    if not val:
        return JSONResponse(status_code=404, content={'detail': 'Result not found'})
    return JSONResponse(status_code=200, content=val)




        
      