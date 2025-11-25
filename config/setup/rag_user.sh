USER_=$(whoami)
echo "Current OS User: $USER_"

psql -U $USER_ -d postgres -f /config/setup/rag_user.sql
# psql -U $USER_ -d postgres -f ./rag_user.sql
