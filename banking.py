# Banking App with Microservices Architecture and E-commerce Platform
# Complete system with multiple services

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import jwt
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import uvicorn
from threading import Thread
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS
# =============================================================================

class AccountType(Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    BUSINESS = "business"

class TransactionType(Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    REFUND = "refund"

class TransactionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

@dataclass
class User:
    user_id: str
    email: str
    first_name: str
    last_name: str
    phone: str
    created_at: datetime
    is_active: bool = True
    kyc_verified: bool = False

@dataclass
class Account:
    account_id: str
    user_id: str
    account_type: AccountType
    balance: float
    account_number: str
    created_at: datetime
    is_active: bool = True

@dataclass
class Transaction:
    transaction_id: str
    from_account: Optional[str]
    to_account: Optional[str]
    amount: float
    transaction_type: TransactionType
    status: TransactionStatus
    description: str
    created_at: datetime
    updated_at: datetime

@dataclass
class Product:
    product_id: str
    name: str
    description: str
    price: float
    category: str
    stock: int
    merchant_id: str
    created_at: datetime

@dataclass
class Order:
    order_id: str
    user_id: str
    products: List[Dict]
    total_amount: float
    status: OrderStatus
    payment_method: str
    shipping_address: Dict
    created_at: datetime
    updated_at: datetime

# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    phone: str
    password: str

class AccountCreate(BaseModel):
    account_type: str
    initial_deposit: float = 0.0

class TransactionCreate(BaseModel):
    from_account: Optional[str] = None
    to_account: Optional[str] = None
    amount: float
    transaction_type: str
    description: str

class ProductCreate(BaseModel):
    name: str
    description: str
    price: float
    category: str
    stock: int

class OrderCreate(BaseModel):
    products: List[Dict]
    payment_method: str
    shipping_address: Dict

# =============================================================================
# DATABASE LAYER (In-Memory for Demo)
# =============================================================================

class DatabaseManager:
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.accounts: Dict[str, Account] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.products: Dict[str, Product] = {}
        self.orders: Dict[str, Order] = {}
        self.user_passwords: Dict[str, str] = {}
        self.user_sessions: Dict[str, str] = {}

    def save_user(self, user: User, password: str) -> bool:
        self.users[user.user_id] = user
        self.user_passwords[user.user_id] = self._hash_password(password)
        return True

    def get_user(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    def verify_password(self, user_id: str, password: str) -> bool:
        stored_hash = self.user_passwords.get(user_id)
        return stored_hash == self._hash_password(password)

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def save_account(self, account: Account) -> bool:
        self.accounts[account.account_id] = account
        return True

    def get_account(self, account_id: str) -> Optional[Account]:
        return self.accounts.get(account_id)

    def get_user_accounts(self, user_id: str) -> List[Account]:
        return [acc for acc in self.accounts.values() if acc.user_id == user_id]

    def update_balance(self, account_id: str, new_balance: float) -> bool:
        if account_id in self.accounts:
            self.accounts[account_id].balance = new_balance
            return True
        return False

    def save_transaction(self, transaction: Transaction) -> bool:
        self.transactions[transaction.transaction_id] = transaction
        return True

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        return self.transactions.get(transaction_id)

    def get_account_transactions(self, account_id: str) -> List[Transaction]:
        return [
            txn for txn in self.transactions.values()
            if txn.from_account == account_id or txn.to_account == account_id
        ]

    def save_product(self, product: Product) -> bool:
        self.products[product.product_id] = product
        return True

    def get_product(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)

    def get_all_products(self) -> List[Product]:
        return list(self.products.values())

    def save_order(self, order: Order) -> bool:
        self.orders[order.order_id] = order
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_user_orders(self, user_id: str) -> List[Order]:
        return [order for order in self.orders.values() if order.user_id == user_id]

# =============================================================================
# AUTHENTICATION SERVICE
# =============================================================================

class AuthService:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.secret_key = "your-secret-key-here"
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def authenticate_user(self, email: str, password: str) -> Optional[str]:
        user = self.db.get_user_by_email(email)
        if not user or not self.db.verify_password(user.user_id, password):
            return None
        
        token_data = {
            "user_id": user.user_id,
            "email": user.email,
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        }
        
        token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Optional[str]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("user_id")
            if user_id is None:
                return None
            return user_id
        except jwt.PyJWTError:
            return None

# =============================================================================
# CORE SERVICES
# =============================================================================

class UserService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    async def create_user(self, user_data: UserCreate) -> User:
        # Check if user already exists
        if self.db.get_user_by_email(user_data.email):
            raise HTTPException(status_code=400, detail="Email already registered")

        user = User(
            user_id=str(uuid.uuid4()),
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            phone=user_data.phone,
            created_at=datetime.utcnow()
        )

        self.db.save_user(user, user_data.password)
        logger.info(f"User created: {user.email}")
        return user

    async def get_user_profile(self, user_id: str) -> User:
        user = self.db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

class AccountService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    async def create_account(self, user_id: str, account_data: AccountCreate) -> Account:
        user = self.db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        account = Account(
            account_id=str(uuid.uuid4()),
            user_id=user_id,
            account_type=AccountType(account_data.account_type),
            balance=account_data.initial_deposit,
            account_number=f"ACC{int(time.time())}{uuid.uuid4().hex[:6].upper()}",
            created_at=datetime.utcnow()
        )

        self.db.save_account(account)
        
        # Create initial deposit transaction if amount > 0
        if account_data.initial_deposit > 0:
            transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                from_account=None,
                to_account=account.account_id,
                amount=account_data.initial_deposit,
                transaction_type=TransactionType.DEPOSIT,
                status=TransactionStatus.COMPLETED,
                description="Initial deposit",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.db.save_transaction(transaction)

        logger.info(f"Account created: {account.account_number}")
        return account

    async def get_user_accounts(self, user_id: str) -> List[Account]:
        return self.db.get_user_accounts(user_id)

    async def get_account_balance(self, account_id: str, user_id: str) -> float:
        account = self.db.get_account(account_id)
        if not account or account.user_id != user_id:
            raise HTTPException(status_code=404, detail="Account not found")
        return account.balance

class TransactionService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    async def process_transaction(self, transaction_data: TransactionCreate) -> Transaction:
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            from_account=transaction_data.from_account,
            to_account=transaction_data.to_account,
            amount=transaction_data.amount,
            transaction_type=TransactionType(transaction_data.transaction_type),
            status=TransactionStatus.PENDING,
            description=transaction_data.description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        # Process based on transaction type
        try:
            if transaction.transaction_type == TransactionType.DEPOSIT:
                await self._process_deposit(transaction)
            elif transaction.transaction_type == TransactionType.WITHDRAWAL:
                await self._process_withdrawal(transaction)
            elif transaction.transaction_type == TransactionType.TRANSFER:
                await self._process_transfer(transaction)
            elif transaction.transaction_type == TransactionType.PAYMENT:
                await self._process_payment(transaction)

            transaction.status = TransactionStatus.COMPLETED
            transaction.updated_at = datetime.utcnow()

        except Exception as e:
            transaction.status = TransactionStatus.FAILED
            transaction.updated_at = datetime.utcnow()
            logger.error(f"Transaction failed: {str(e)}")

        self.db.save_transaction(transaction)
        return transaction

    async def _process_deposit(self, transaction: Transaction):
        if not transaction.to_account:
            raise ValueError("To account required for deposit")
        
        account = self.db.get_account(transaction.to_account)
        if not account:
            raise ValueError("Account not found")
        
        new_balance = account.balance + transaction.amount
        self.db.update_balance(transaction.to_account, new_balance)

    async def _process_withdrawal(self, transaction: Transaction):
        if not transaction.from_account:
            raise ValueError("From account required for withdrawal")
        
        account = self.db.get_account(transaction.from_account)
        if not account:
            raise ValueError("Account not found")
        
        if account.balance < transaction.amount:
            raise ValueError("Insufficient funds")
        
        new_balance = account.balance - transaction.amount
        self.db.update_balance(transaction.from_account, new_balance)

    async def _process_transfer(self, transaction: Transaction):
        if not transaction.from_account or not transaction.to_account:
            raise ValueError("Both accounts required for transfer")
        
        from_account = self.db.get_account(transaction.from_account)
        to_account = self.db.get_account(transaction.to_account)
        
        if not from_account or not to_account:
            raise ValueError("Account not found")
        
        if from_account.balance < transaction.amount:
            raise ValueError("Insufficient funds")
        
        # Update balances
        from_balance = from_account.balance - transaction.amount
        to_balance = to_account.balance + transaction.amount
        
        self.db.update_balance(transaction.from_account, from_balance)
        self.db.update_balance(transaction.to_account, to_balance)

    async def _process_payment(self, transaction: Transaction):
        # Similar to withdrawal for payments
        await self._process_withdrawal(transaction)

    async def get_account_transactions(self, account_id: str, user_id: str) -> List[Transaction]:
        account = self.db.get_account(account_id)
        if not account or account.user_id != user_id:
            raise HTTPException(status_code=404, detail="Account not found")
        
        return self.db.get_account_transactions(account_id)

class EcommerceService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    async def create_product(self, product_data: ProductCreate, merchant_id: str) -> Product:
        product = Product(
            product_id=str(uuid.uuid4()),
            name=product_data.name,
            description=product_data.description,
            price=product_data.price,
            category=product_data.category,
            stock=product_data.stock,
            merchant_id=merchant_id,
            created_at=datetime.utcnow()
        )
        
        self.db.save_product(product)
        logger.info(f"Product created: {product.name}")
        return product

    async def get_all_products(self) -> List[Product]:
        return self.db.get_all_products()

    async def create_order(self, user_id: str, order_data: OrderCreate) -> Order:
        # Calculate total amount
        total_amount = 0
        for item in order_data.products:
            product = self.db.get_product(item["product_id"])
            if not product:
                raise HTTPException(status_code=404, detail=f"Product {item['product_id']} not found")
            if product.stock < item["quantity"]:
                raise HTTPException(status_code=400, detail=f"Insufficient stock for {product.name}")
            total_amount += product.price * item["quantity"]

        order = Order(
            order_id=str(uuid.uuid4()),
            user_id=user_id,
            products=order_data.products,
            total_amount=total_amount,
            status=OrderStatus.PENDING,
            payment_method=order_data.payment_method,
            shipping_address=order_data.shipping_address,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.db.save_order(order)
        logger.info(f"Order created: {order.order_id}")
        return order

    async def process_payment(self, order_id: str, account_id: str) -> bool:
        order = self.db.get_order(order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")

        account = self.db.get_account(account_id)
        if not account or account.user_id != order.user_id:
            raise HTTPException(status_code=404, detail="Account not found")

        if account.balance < order.total_amount:
            raise HTTPException(status_code=400, detail="Insufficient funds")

        # Process payment transaction
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            from_account=account_id,
            to_account=None,  # Merchant account would be here
            amount=order.total_amount,
            transaction_type=TransactionType.PAYMENT,
            status=TransactionStatus.COMPLETED,
            description=f"Payment for order {order_id}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        # Update account balance
        new_balance = account.balance - order.total_amount
        self.db.update_balance(account_id, new_balance)
        self.db.save_transaction(transaction)

        # Update order status
        order.status = OrderStatus.CONFIRMED
        order.updated_at = datetime.utcnow()
        self.db.save_order(order)

        # Update product stock
        for item in order.products:
            product = self.db.get_product(item["product_id"])
            if product:
                product.stock -= item["quantity"]
                self.db.save_product(product)

        return True

    async def get_user_orders(self, user_id: str) -> List[Order]:
        return self.db.get_user_orders(user_id)

# =============================================================================
# API GATEWAY / MAIN APPLICATION
# =============================================================================

class BankingApp:
    def __init__(self):
        self.db = DatabaseManager()
        self.auth_service = AuthService(self.db)
        self.user_service = UserService(self.db)
        self.account_service = AccountService(self.db)
        self.transaction_service = TransactionService(self.db)
        self.ecommerce_service = EcommerceService(self.db)
        
        # Setup FastAPI app
        self.app = FastAPI(title="Banking & E-commerce Platform", version="1.0.0")
        self._setup_routes()
        
        # Setup security
        self.security = HTTPBearer()

    def _setup_routes(self):
        # Authentication routes
        @self.app.post("/auth/register")
        async def register(user_data: UserCreate):
            user = await self.user_service.create_user(user_data)
            return {"message": "User created successfully", "user_id": user.user_id}

        @self.app.post("/auth/login")
        async def login(email: str, password: str):
            token = self.auth_service.authenticate_user(email, password)
            if not token:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            return {"access_token": token, "token_type": "bearer"}

        # User routes
        @self.app.get("/users/profile")
        async def get_profile(current_user: str = Depends(self.get_current_user)):
            user = await self.user_service.get_user_profile(current_user)
            return asdict(user)

        # Account routes
        @self.app.post("/accounts")
        async def create_account(
            account_data: AccountCreate,
            current_user: str = Depends(self.get_current_user)
        ):
            account = await self.account_service.create_account(current_user, account_data)
            return asdict(account)

        @self.app.get("/accounts")
        async def get_accounts(current_user: str = Depends(self.get_current_user)):
            accounts = await self.account_service.get_user_accounts(current_user)
            return [asdict(acc) for acc in accounts]

        @self.app.get("/accounts/{account_id}/balance")
        async def get_balance(
            account_id: str,
            current_user: str = Depends(self.get_current_user)
        ):
            balance = await self.account_service.get_account_balance(account_id, current_user)
            return {"balance": balance}

        # Transaction routes
        @self.app.post("/transactions")
        async def create_transaction(
            transaction_data: TransactionCreate,
            current_user: str = Depends(self.get_current_user)
        ):
            transaction = await self.transaction_service.process_transaction(transaction_data)
            return asdict(transaction)

        @self.app.get("/accounts/{account_id}/transactions")
        async def get_transactions(
            account_id: str,
            current_user: str = Depends(self.get_current_user)
        ):
            transactions = await self.transaction_service.get_account_transactions(account_id, current_user)
            return [asdict(txn) for txn in transactions]

        # E-commerce routes
        @self.app.post("/products")
        async def create_product(
            product_data: ProductCreate,
            current_user: str = Depends(self.get_current_user)
        ):
            product = await self.ecommerce_service.create_product(product_data, current_user)
            return asdict(product)

        @self.app.get("/products")
        async def get_products():
            products = await self.ecommerce_service.get_all_products()
            return [asdict(product) for product in products]

        @self.app.post("/orders")
        async def create_order(
            order_data: OrderCreate,
            current_user: str = Depends(self.get_current_user)
        ):
            order = await self.ecommerce_service.create_order(current_user, order_data)
            return asdict(order)

        @self.app.post("/orders/{order_id}/pay")
        async def pay_order(
            order_id: str,
            account_id: str,
            current_user: str = Depends(self.get_current_user)
        ):
            success = await self.ecommerce_service.process_payment(order_id, account_id)
            return {"success": success, "message": "Payment processed successfully"}

        @self.app.get("/orders")
        async def get_orders(current_user: str = Depends(self.get_current_user)):
            orders = await self.ecommerce_service.get_user_orders(current_user)
            return [asdict(order) for order in orders]

        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow()}

    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        user_id = self.auth_service.verify_token(credentials.credentials)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user_id

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

# =============================================================================
# DEMO DATA AND TESTING
# =============================================================================

async def create_demo_data(banking_app: BankingApp):
    """Create some demo data for testing"""
    
    # Create demo users
    demo_users = [
        UserCreate(
            email="alice@example.com",
            first_name="Alice",
            last_name="Johnson",
            phone="+1234567890",
            password="password123"
        ),
        UserCreate(
            email="bob@example.com",
            first_name="Bob",
            last_name="Smith",
            phone="+1234567891",
            password="password123"
        )
    ]
    
    users = []
    for user_data in demo_users:
        try:
            user = await banking_app.user_service.create_user(user_data)
            users.append(user)
            print(f"Created user: {user.email}")
        except:
            pass
    
    # Create demo accounts
    for user in users:
        account_data = AccountCreate(account_type="checking", initial_deposit=1000.0)
        account = await banking_app.account_service.create_account(user.user_id, account_data)
        print(f"Created account: {account.account_number}")
    
    # Create demo products
    demo_products = [
        ProductCreate(
            name="Laptop",
            description="High-performance laptop",
            price=999.99,
            category="Electronics",
            stock=10
        ),
        ProductCreate(
            name="Coffee Mug",
            description="Ceramic coffee mug",
            price=15.99,
            category="Home & Kitchen",
            stock=50
        )
    ]
    
    for product_data in demo_products:
        product = await banking_app.ecommerce_service.create_product(product_data, users[0].user_id)
        print(f"Created product: {product.name}")

def main():
    """Main function to run the banking application"""
    print("🏦 Starting Banking & E-commerce Platform...")
    
    # Create banking app instance
    banking_app = BankingApp()
    
    # Create demo data in a separate thread
    async def setup_demo():
        await asyncio.sleep(1)  # Wait for app to start
        await create_demo_data(banking_app)
        print("\n✅ Demo data created successfully!")
        print("\n📚 API Documentation available at: http://localhost:8000/docs")
        print("\n🔐 Demo Users:")
        print("   - alice@example.com / password123")
        print("   - bob@example.com / password123")
        print("\n🚀 Try the following workflow:")
        print("   1. POST /auth/login to get token")
        print("   2. GET /accounts to see demo accounts")
        print("   3. GET /products to see demo products")
        print("   4. POST /orders to create an order")
        print("   5. POST /orders/{order_id}/pay to pay for order")
    
    # Run demo setup in background
    def run_demo_setup():
        asyncio.run(setup_demo())
    
    demo_thread = Thread(target=run_demo_setup)
    demo_thread.daemon = True
    demo_thread.start()
    
    # Start the FastAPI application
    banking_app.run()

if __name__ == "__main__":
    main()