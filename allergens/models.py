from django.db import models


class Users(models.Model):
    user_id = models.TextField(db_column='user_Id', primary_key=True)
    gmail = models.EmailField(db_column='Gmail')
    fname = models.TextField(db_column='Fname', blank=True, null=True)
    lname = models.TextField(db_column='Lname', blank=True, null=True)
    age = models.DecimalField(max_digits=3, decimal_places=0, blank=True, null=True)
    height = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    weight = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    activity = models.TextField(blank=True, null=True)
    diet = models.TextField(blank=True, null=True)
    lifestyle = models.TextField(blank=True, null=True)
    disease = models.TextField(blank=True, null=True)
    image = models.TextField(blank=True, null=True)
    sugar = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    bp = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    cholestrol = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    heartrate = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)
    bmi = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'Users'


class History(models.Model):
    user = models.ForeignKey('Users', models.DO_NOTHING, db_column='user_Id')
    code = models.DecimalField(max_digits=10, decimal_places=2)
    brandname = models.TextField(db_column='brandName', blank=True, null=True)
    name = models.TextField(blank=True, null=True)
    image = models.TextField(blank=True, null=True)
    slno = models.UUIDField(primary_key=True)
    ingredients = models.TextField(blank=True, null=True)
    nutrients = models.TextField(blank=True, null=True)
    score = models.TextField(blank=True, null=True)
    nutri = models.TextField(db_column='Nutri', blank=True, null=True)
    allergens = models.TextField(blank=True, null=True)
    hazard = models.TextField(blank=True, null=True)
    long = models.TextField(db_column='Long', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'History'
        db_table_comment = 'This is a duplicate of Saved'


class Saved(models.Model):
    user = models.ForeignKey(Users, models.DO_NOTHING, db_column='user_Id')
    code = models.DecimalField(max_digits=10, decimal_places=2)
    brandname = models.TextField(db_column='brandName', blank=True, null=True)
    name = models.TextField(blank=True, null=True)
    image = models.TextField(blank=True, null=True)
    slno = models.UUIDField(primary_key=True)
    ingredients = models.TextField(blank=True, null=True)
    nutrients = models.TextField(blank=True, null=True)
    score = models.TextField(blank=True, null=True)
    nutri = models.TextField(db_column='Nutri', blank=True, null=True)
    allergens = models.TextField(blank=True, null=True)
    hazard = models.TextField(blank=True, null=True)
    long = models.TextField(db_column='Long', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'Saved'
        db_table_comment = 'This is a duplicate of History'
